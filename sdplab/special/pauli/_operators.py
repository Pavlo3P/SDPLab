"""Symbolic Pauli-string operators and Pauli-basis linear combinations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from numbers import Number
from typing import Iterable, Sequence, Tuple, Any
from itertools import product
import math

from spacecore import Context, DenseArray, jax_pytree_class, BackendOps, ContextBound, resolve_context_priority
from spacecore._contextual import normalize_context

from ._pauli import pauli_matrices, _PAULI_TO_CODE, _CODE_TO_PAULI, _MUL_TABLE


def _validate_identifier(identifier: str) -> str:
    """Validate and normalize a Pauli identifier string."""
    if not isinstance(identifier, str):
        raise TypeError(f"identifier must be str, got {type(identifier)!r}")
    if len(identifier) == 0:
        raise ValueError("identifier must be non-empty")
    identifier = identifier.upper()
    bad = sorted({ch for ch in identifier if ch not in _PAULI_TO_CODE})
    if bad:
        raise ValueError(
            f"Invalid Pauli characters {bad!r}. Expected only characters from 'IXYZ'."
        )
    return identifier


def _codes_from_identifier(identifier: str) -> Tuple[int, ...]:
    """Encode a Pauli identifier as integer Pauli codes."""
    return tuple(_PAULI_TO_CODE[ch] for ch in identifier)


def _identifier_from_codes(codes: Sequence[int]) -> str:
    """Decode integer Pauli codes into a Pauli identifier string."""
    return "".join(_CODE_TO_PAULI[c] for c in codes)


def _perm_to_front(axis: int, ndim: int) -> Tuple[int, ...]:
    """Return a permutation that moves ``axis`` to the front."""
    return (axis,) + tuple(i for i in range(ndim) if i != axis)


def _inverse_perm(perm: Sequence[int]) -> Tuple[int, ...]:
    """Return the inverse of a tensor-axis permutation."""
    out = [0] * len(perm)
    for i, p in enumerate(perm):
        out[p] = i
    return tuple(out)


def _apply_one_site(X: DenseArray, code: int, axis: int, ops: BackendOps) -> DenseArray:
    """Apply one encoded Pauli factor to one axis of a qubit tensor."""
    if code == 0:
        return X

    perm = _perm_to_front(axis, len(X.shape))
    inv_perm = _inverse_perm(perm)
    Xp = ops.transpose(X, perm)
    x0 = Xp[0]
    x1 = Xp[1]

    if code == 1:  # X
        Yp = ops.stack([x1, x0], axis=0)
    elif code == 2:  # Y
        Yp = ops.stack([-1j * x1, 1j * x0], axis=0)
    elif code == 3:  # Z
        Yp = ops.stack([x0, -x1], axis=0)
    else:
        raise ValueError(f"Unknown Pauli code {code!r}")

    return ops.transpose(Yp, inv_perm)


def _apply_codes_to_vector(codes: Sequence[int], x: DenseArray, ops: BackendOps, phase: complex = 1.0) -> DenseArray:
    """Apply a full Pauli code word to a state vector without materializing it."""
    n_qubits = len(codes)
    expected_shape = (2 ** n_qubits,)
    if tuple(x.shape) != expected_shape:
        raise ValueError(f"Expected vector shape {expected_shape}, got {tuple(x.shape)}")

    X = ops.reshape(x, (2,) * n_qubits)
    Y = X
    for axis, code in enumerate(codes):
        Y = _apply_one_site(Y, code, axis=axis, ops=ops)
    y = ops.reshape(Y, expected_shape)
    return phase * y if phase != 1 else y


def _combine_codes(left: Sequence[int], right: Sequence[int]) -> Tuple[complex, Tuple[int, ...]]:
    """Multiply two encoded Pauli strings and return phase plus output codes."""
    if len(left) != len(right):
        raise ValueError(
            f"Pauli strings must have the same length, got {len(left)} and {len(right)}"
        )

    phase = 1.0 + 0.0j
    out = []
    for lc, rc in zip(left, right):
        local_phase, code = _MUL_TABLE[(int(lc), int(rc))]
        phase *= local_phase
        out.append(code)
    return phase, tuple(out)


def _apply_one_site_mat(
    X: DenseArray,
    code: int,
    axis: int,
    ops: BackendOps,
) -> DenseArray:
    """
    Apply a single-qubit Pauli operator on a chosen tensor axis of X.

    Parameters
    ----------
    X:
        Tensor of shape (2, ..., 2, k) or more generally with a qubit axis of
        size 2 at position `axis`.
    code:
        Encoded Pauli: 0 -> I, 1 -> X, 2 -> Y, 3 -> Z.
    axis:
        Qubit axis on which to act.
    ops:
        Backend ops object.

    Returns
    -------
    DenseArray
        Tensor with the same shape as X.
    """
    if code == 0:
        return X

    M = ops.asarray(pauli_matrices[_CODE_TO_PAULI[code]])

    ndim = len(X.shape)
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if ndim + 1 > len(letters):
        raise ValueError("Tensor rank too large for einsum index construction.")

    in_idx = list(letters[:ndim])
    out_idx = in_idx.copy()

    contracted = in_idx[axis]
    new_idx = letters[ndim]

    # M has indices (new_idx, contracted)
    out_idx[axis] = new_idx

    eq = f"{new_idx}{contracted},{''.join(in_idx)}->{''.join(out_idx)}"
    return ops.einsum(eq, M, X)


@jax_pytree_class
@dataclass(init=False)
class PauliString(ContextBound):
    r"""Symbolic tensor-product Pauli operator.

    A label such as ``"IXZ"`` represents
    :math:`I \otimes X \otimes Z` acting on three qubits. The optional complex
    ``phase`` represents the scalar multiplier, so the stored operator is
    :math:`\mathrm{phase}\,P_0 \otimes \cdots \otimes P_{n-1}`. The dense
    matrix can be materialized, but vector and matrix application use tensor
    structure when possible.
    """

    def __init__(
        self,
        identifier: str,
        *,
        phase: complex = 1.0,
        ctx: Context | str | None = None,
    ) -> None:
        """Create ``phase * P_0 otimes ... otimes P_{n-1}`` from a label."""
        super(PauliString, self).__init__(ctx)
        identifier = _validate_identifier(identifier)
        self.identifier = identifier
        self.codes = _codes_from_identifier(identifier)
        self.phase = complex(phase)

    @property
    def n_qubits(self) -> int:
        """Return ``n`` for an operator on ``(C^2)^{otimes n}``."""
        return len(self.codes)

    @property
    def label(self) -> str:
        """Return the normalized string label, such as ``"IXZ"``."""
        return self.identifier

    def copy(self) -> PauliString:
        """Return an independent copy with the same label, phase, and context."""
        return PauliString(self.identifier, phase=self.phase, ctx=self.ctx)

    def support(self) -> Tuple[int, ...]:
        r"""
        Return the support of the Pauli operator.

        The support is the tuple of qubit indices on which the operator acts
        nontrivially, i.e. the sites where the local Pauli factor is not the
        identity.

        .. math::

            P = P_0 \otimes P_1 \otimes \cdots \otimes P_{n-1},

        with each local factor :math:`P_k \in \{I, X, Y, Z\}`, then

        .. math::

            \operatorname{support}(P) = \{k : P_k \ne I\}.

        Returns:
            tuple[int, ...]:
                Sorted tuple of qubit indices where the local factor is not
                identity.

        Examples:
            "IIXZI" has support (2, 3) with zero-based indexing.
            "IIII" has empty support ().

        Notes:
            The size of the support is the Pauli weight, i.e.
            ``len(self.support()) == self.weight()``.
        """
        return tuple(i for i, c in enumerate(self.codes) if c != 0)

    def weight(self) -> int:
        r"""
        Return the Pauli weight of the operator.

        The Pauli weight is the number of qubit sites on which the operator acts
        nontrivially, i.e. the number of local factors that are not the identity.

        .. math::

            P = P_0 \otimes P_1 \otimes \cdots \otimes P_{n-1},

        with each local factor :math:`P_k \in \{I, X, Y, Z\}`, then

        .. math::

            \operatorname{weight}(P) = |\{k : P_k \ne I\}|.

        Returns:
            int:
                Number of qubit indices where the local factor is not identity.

        Examples:
            "IIXZI" has weight 2.
            "IIII" has weight 0.

        Notes:
            The Pauli weight is the cardinality of the support, so
            ``self.weight() == len(self.support())``.
        """
        return len(self.support())

    def is_identity(self) -> bool:
        """Return True if every local Pauli factor is the identity."""
        return all(c == 0 for c in self.codes)

    def adjoint(self) -> PauliString:
        """Return the Hermitian adjoint of this symbolic Pauli string."""
        return PauliString(self.identifier, phase=self.phase.conjugate(), ctx=self.ctx)

    dagger = adjoint

    def trace(self):
        """Return the trace of the represented dense Pauli operator."""
        if self.is_identity():
            return self.phase * (2 ** self.n_qubits)
        return self.phase * 0.0

    def commutes_with(self, other: PauliString) -> bool:
        """Return True when this Pauli string commutes with ``other``."""
        self._check_compatible(other)
        anticomm_sites = 0
        for lc, rc in zip(self.codes, other.codes):
            if lc != 0 and rc != 0 and lc != rc:
                anticomm_sites += 1
        return (anticomm_sites % 2) == 0

    def multiply(self, other: PauliString) -> PauliString:
        """Return the symbolic product of two compatible Pauli strings."""
        self._check_compatible(other)
        local_phase, codes = _combine_codes(self.codes, other.codes)
        return PauliString(
            _identifier_from_codes(codes),
            phase=self.phase * other.phase * local_phase,
            ctx=self.ctx,
        )

    def materialize(self) -> DenseArray:
        """Return the full ``2^n x 2^n`` matrix for this Pauli string."""
        mats = [self.ctx.ops.asarray(pauli_matrices[ch]) for ch in self.identifier]
        out = reduce(self.ctx.ops.kron, mats)
        return self.phase * out if self.phase != 1 else out

    def matvec(self, x: DenseArray) -> DenseArray:
        """Return ``P x`` for a state vector ``x`` of length ``2^n``."""
        x = self.ctx.ops.asarray(x)
        return _apply_codes_to_vector(self.codes, x, self.ctx.ops, phase=self.phase)

    apply = matvec

    def to_sum(self, coeff: complex = 1.0) -> PauliSum:
        """Represent this Pauli string as a one-term ``PauliSum``."""
        return PauliSum([self], coeffs=[coeff], ctx=self.ctx)

    def _check_compatible(self, other: PauliString) -> None:
        if not isinstance(other, PauliString):
            raise TypeError(f"Expected PauliString, got {type(other)!r}")
        if self.n_qubits != other.n_qubits:
            raise ValueError(
                f"Incompatible Pauli lengths: {self.n_qubits} and {other.n_qubits}"
            )

    def __matmul__(self, other):
        if isinstance(other, PauliString):
            return self.multiply(other)

        other = self.ctx.ops.asarray(other)
        if other.ndim == 1:
            return self.matvec(other)
        if other.ndim == 2:
            return self.matmat(other)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Number):
            return self.to_sum(coeff=other)
        if isinstance(other, PauliString):
            return self.multiply(other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Number):
            return self.to_sum(coeff=other)
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, PauliString):
            return PauliSum([self, other], ctx=self.ctx)
        if isinstance(other, PauliSum):
            return other + self
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, PauliString):
            return PauliSum([self, other], coeffs=[1.0, -1.0], ctx=self.ctx)
        if isinstance(other, PauliSum):
            return self.to_sum() - other
        return NotImplemented

    def tree_flatten(self):
        """Return children and auxiliary data for JAX PyTree flattening."""
        return (), (self.identifier, self.phase, self.ctx)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild a Pauli string from JAX PyTree data."""
        identifier, phase, ctx = aux
        return cls(identifier, phase=phase, ctx=ctx)

    def _convert(self, ctx: Context) -> PauliString:
        return PauliString(self.identifier, phase=self.phase, ctx=ctx)

    def matmat(self, X: DenseArray) -> DenseArray:
        """
        Left-multiply a matrix by this Pauli string.

        If this Pauli string acts on n qubits, and X has shape (2**n, k), this
        returns P X without materializing the full dense matrix P.

        Args:
            X:
                Dense matrix of shape (2**n, k).

        Returns:
            DenseArray:
                Matrix of shape (2**n, k).
        """
        X = self.ctx.ops.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"`X` must have ndim 2, got shape {X.shape}")

        expected_rows = 2 ** self.n_qubits
        if X.shape[0] != expected_rows:
            raise ValueError(
                f"Expected X.shape[0] == {expected_rows}, got {X.shape[0]}"
            )

        k = X.shape[1]
        Y = self.ctx.ops.reshape(X, (2,) * self.n_qubits + (k,))

        for axis, code in enumerate(self.codes):
            Y = _apply_one_site_mat(Y, code, axis=axis, ops=self.ctx.ops)

        Y = self.ctx.ops.reshape(Y, (expected_rows, k))
        return self.phase * Y if self.phase != 1 else Y


@jax_pytree_class
@dataclass(init=False)
class PauliSum(ContextBound):
    r"""Symbolic linear combination of Pauli strings.

    Represents an operator

    .. math::

        A = \sum_j c_j P_j

    where every ``P_j`` acts on the same ``n``-qubit Hilbert space. Repeated
    Pauli labels may be merged into one coefficient when ``simplify=True``.
    """

    def __init__(
        self,
        terms: Iterable[str | PauliString],
        coeffs: Iterable[complex] | None = None,
        *,
        ctx: Context | str | None = None,
        simplify: bool = True,
    ) -> None:
        """Create ``sum_j coeffs[j] * terms[j]`` on a common qubit register."""
        raw_terms = list(terms)
        if not raw_terms:
            raise ValueError("terms must be non-empty")

        term_objs = [
            term if isinstance(term, PauliString) else PauliString(term, ctx=ctx)
            for term in raw_terms
        ]
        resolved_ctx = resolve_context_priority(ctx, *term_objs)
        super(PauliSum, self).__init__(resolved_ctx)

        n_qubits = term_objs[0].n_qubits
        if coeffs is None:
            coeffs_list = [1.0] * len(term_objs)
        else:
            coeffs_list = list(coeffs)
            if len(coeffs_list) != len(term_objs):
                raise ValueError(
                    f"coeffs and terms must have the same length, got {len(coeffs_list)} and {len(term_objs)}"
                )

        normalized_terms: list[PauliString] = []
        normalized_coeffs: list[complex] = []
        for coeff, term in zip(coeffs_list, term_objs):
            if term.n_qubits != n_qubits:
                raise ValueError(
                    f"All Pauli terms must have the same length, got {n_qubits} and {term.n_qubits}"
                )
            normalized_terms.append(PauliString(term.identifier, phase=1.0, ctx=self.ctx))
            normalized_coeffs.append(complex(coeff) * complex(term.phase))

        if simplify:
            normalized_terms, normalized_coeffs = self._simplify_terms(
                normalized_terms,
                normalized_coeffs,
            )

        self._n_qubits = n_qubits
        self.terms = tuple(normalized_terms)
        self.codes = tuple(term.codes for term in self.terms)
        self.coeffs = self.ctx.ops.asarray(normalized_coeffs)

    @property
    def n_qubits(self) -> int:
        """Return ``n`` for an operator on ``(C^2)^{otimes n}``."""
        return self._n_qubits

    @property
    def n_terms(self) -> int:
        """Return the number of symbolic Pauli terms in the sum."""
        return len(self.terms)

    def support(self) -> Tuple[int, ...]:
        r"""
        Return the support of the Pauli sum.

        The support is the tuple of qubit indices on which at least one Pauli
        term acts nontrivially. Equivalently, it is the union of the supports
        of all Pauli-string terms in the sum.

        .. math::

            A = \sum_j c_j P_j,

        then

        .. math::

            \operatorname{support}(A) =
            \bigcup_j \operatorname{support}(P_j).

        Returns:
            tuple[int, ...]:
                Sorted tuple of qubit indices touched nontrivially by at least
                one term.

        Notes:
            If the sum has no terms, the support is empty.
        """
        sites = set()
        for term in self.terms:
            sites.update(term.support())
        return tuple(sorted(sites))

    def simplify(self) -> PauliSum:
        """Return a new sum with duplicate labels merged."""
        return PauliSum(self.terms, coeffs=self.coeffs, ctx=self.ctx, simplify=True)

    def materialize(self) -> DenseArray:
        """Return the full dense matrix ``sum_j c_j P_j``."""
        out = None
        for coeff, term in zip(self.coeffs, self.terms):
            term_mat = coeff * term.materialize()
            out = term_mat if out is None else out + term_mat
        return out

    def matvec(self, x: DenseArray) -> DenseArray:
        """Return ``sum_j c_j P_j x`` without materializing the full matrix."""
        x = self.ctx.ops.asarray(x)
        acc = x * 0
        for coeff, term in zip(self.coeffs, self.terms):
            acc = acc + coeff * _apply_codes_to_vector(term.codes, x, self.ctx.ops, phase=term.phase)
        return acc

    apply = matvec

    def add_term(self, term: str | PauliString, coeff: complex = 1.0) -> "PauliSum":
        """Return a new sum with one additional term and coefficient."""
        if not isinstance(term, PauliString):
            term = PauliString(term, ctx=self.ctx)
        return PauliSum(
            [*self.terms, term],
            coeffs=[*list(self.coeffs), coeff],
            ctx=self.ctx,
            simplify=True,
        )

    def trace(self):
        """Return the trace of the represented dense operator."""
        total = 0.0 + 0.0j
        for coeff, term in zip(self.coeffs, self.terms):
            total += coeff * term.trace()
        return total

    def __matmul__(self, other):
        if isinstance(other, PauliString):
            new_terms = [term.multiply(other) for term in self.terms]
            return PauliSum(new_terms, coeffs=self.coeffs, ctx=self.ctx, simplify=True)

        if isinstance(other, PauliSum):
            new_terms = []
            new_coeffs = []
            for coeff_l, term_l in zip(self.coeffs, self.terms):
                for coeff_r, term_r in zip(other.coeffs, other.terms):
                    new_terms.append(term_l.multiply(term_r))
                    new_coeffs.append(coeff_l * coeff_r)
            return PauliSum(new_terms, coeffs=new_coeffs, ctx=self.ctx, simplify=True)

        other = self.ctx.ops.asarray(other)
        if other.ndim == 1:
            return self.matvec(other)
        if other.ndim == 2:
            return self.matmat(other)
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, PauliString):
            return PauliSum([*self.terms, other], coeffs=[*list(self.coeffs), 1.0], ctx=self.ctx, simplify=True)
        if isinstance(other, PauliSum):
            return PauliSum(
                [*self.terms, *other.terms],
                coeffs=[*list(self.coeffs), *list(other.coeffs)],
                ctx=self.ctx,
                simplify=True,
            )
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, PauliString):
            return self + ((-1.0) * other)
        if isinstance(other, PauliSum):
            return self + ((-1.0) * other)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Number):
            return PauliSum(self.terms, coeffs=[other * c for c in self.coeffs], ctx=self.ctx, simplify=False)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Number):
            return self * other
        return NotImplemented

    def tree_flatten(self):
        """Return children and auxiliary data for JAX PyTree flattening."""
        return (self.coeffs,), (self.terms, self.ctx)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild a Pauli sum from JAX PyTree data."""
        (coeffs,) = children
        terms, ctx = aux
        return cls(terms, coeffs=coeffs, ctx=ctx, simplify=False)

    @staticmethod
    def _simplify_terms(
        terms: Sequence[PauliString],
        coeffs: Sequence[complex],
        tol: float = 0.0,
    ) -> tuple[list[PauliString], list[complex]]:
        merged: dict[str, complex] = {}
        for term, coeff in zip(terms, coeffs):
            key = term.identifier
            merged[key] = merged.get(key, 0.0 + 0.0j) + complex(coeff) * complex(term.phase)

        out_terms: list[PauliString] = []
        out_coeffs: list[complex] = []
        for identifier, coeff in sorted(merged.items(), key=lambda item: item[0]):
            if abs(coeff) <= tol:
                continue
            out_terms.append(PauliString(identifier, phase=1.0, ctx=terms[0].ctx))
            out_coeffs.append(coeff)

        if not out_terms:
            out_terms = [PauliString("I" * terms[0].n_qubits, ctx=terms[0].ctx)]
            out_coeffs = [0.0]
        return out_terms, out_coeffs

    def _convert(self, ctx: Context) -> PauliSum:
        return PauliSum(self.terms, coeffs=self.coeffs, ctx=self.ctx, simplify=False)

    @classmethod
    def from_matrix(
        cls,
        mat: Any,
        *,
        tol: float = 1e-12,
        ctx: Context | str | None = None,
        check_hermitian: bool = True,
    ) -> PauliSum:
        r"""
        Decompose a Hermitian matrix into the Pauli-string basis.

        For a matrix A acting on n qubits, this method computes coefficients
        :math:`c_\alpha` such that

        .. math::

            A = \sum_\alpha c_\alpha P_\alpha,

        where the sum runs over all n-qubit Pauli strings and

        .. math::

            c_\alpha = 2^{-n} \operatorname{Tr}[P_\alpha A].

        Args:
            mat:
                Dense matrix of shape (2**n, 2**n).
            tol:
                Terms with :math:`|c_\alpha| \le \mathtt{tol}` are discarded.
            ctx:
                Optional context.
            check_hermitian:
                If True, validate that mat is Hermitian up to numerical
                tolerance.

        Returns:
            PauliSum:
                Symbolic Pauli decomposition of the input matrix.

        Raises:
            ValueError:
                If the input is not square, its dimension is not a power of 2,
                or it is not Hermitian when check_hermitian=True.
        """
        obj_ctx = normalize_context(ctx)
        ops = obj_ctx.ops

        A = ops.asarray(mat)

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(
                f"`mat` must be square, got shape {A.shape}."
            )

        dim = int(A.shape[0])
        if dim <= 0 or (dim & (dim - 1)) != 0:
            raise ValueError(
                f"Matrix dimension must be a positive power of 2, got {dim}."
            )

        n_qubits = int(round(math.log2(dim)))
        if 2 ** n_qubits != dim:
            raise ValueError(
                f"Matrix dimension must equal 2**n for some n, got {dim}."
            )

        labels: list[str] = []
        coeffs: list[complex] = []

        for word in product("IXYZ", repeat=n_qubits):
            label = "".join(word)
            P = PauliString(label, ctx=obj_ctx)
            # coeff = Tr(P A) / 2**n
            coeff = ops.trace(P.materialize() @ A) / dim

            # Hermitian A should give real coefficients in this basis.
            coeff_py = complex(coeff)
            if abs(coeff_py.imag) <= tol:
                coeff_py = coeff_py.real

            if abs(coeff_py) > tol:
                labels.append(label)
                coeffs.append(coeff_py)

        return cls(
            terms=[PauliString(label, ctx=obj_ctx) for label in labels],
            coeffs=coeffs,
            ctx=obj_ctx,
        )

    def matmat(self, X: DenseArray) -> DenseArray:
        r"""
        Left-multiply a matrix by this Pauli sum.

        For

        .. math::

            A = \sum_j c_j P_j,

        this returns

        .. math::

            A X = \sum_j c_j (P_j X),

        without materializing A.

        Args:
            X:
                Dense matrix of shape (2**n, k).

        Returns:
            DenseArray:
                Matrix of shape (2**n, k).
        """
        X = self.ctx.ops.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"`X` must have ndim 2, got shape {X.shape}")

        expected_rows = 2 ** self.n_qubits
        if X.shape[0] != expected_rows:
            raise ValueError(
                f"Expected X.shape[0] == {expected_rows}, got {X.shape[0]}"
            )

        acc = X * 0
        for coeff, term in zip(self.coeffs, self.terms):
            acc = acc + coeff * term.matmat(X)
        return acc
