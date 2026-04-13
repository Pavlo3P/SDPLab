from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from numbers import Number
from typing import Iterable, Sequence, Tuple, Any
from itertools import product
import math

from spacecore import Context, DenseArray, jax_pytree_class, BackendOps
from spacecore._contextual import ContextBound, ctx_manager

from ._pauli import pauli_matrices, _PAULI_TO_CODE, _CODE_TO_PAULI, _MUL_TABLE


def _validate_identifier(identifier: str) -> str:
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
    return tuple(_PAULI_TO_CODE[ch] for ch in identifier)


def _identifier_from_codes(codes: Sequence[int]) -> str:
    return "".join(_CODE_TO_PAULI[c] for c in codes)


def _perm_to_front(axis: int, ndim: int) -> Tuple[int, ...]:
    return (axis,) + tuple(i for i in range(ndim) if i != axis)


def _inverse_perm(perm: Sequence[int]) -> Tuple[int, ...]:
    out = [0] * len(perm)
    for i, p in enumerate(perm):
        out[p] = i
    return tuple(out)


def _apply_one_site(X: DenseArray, code: int, axis: int, ops: BackendOps) -> DenseArray:
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


@jax_pytree_class
@dataclass(init=False)
class PauliString(ContextBound):
    """Symbolic tensor-product Pauli operator with context-aware materialization."""

    def __init__(
        self,
        identifier: str,
        *,
        phase: complex = 1.0,
        ctx: Context | str | None = None,
    ) -> None:
        ctx = ctx_manager.normalize_context(ctx)
        super(PauliString, self).__init__(ctx)
        identifier = _validate_identifier(identifier)
        self.identifier = identifier
        self.codes = _codes_from_identifier(identifier)
        self.phase = complex(phase)

    @property
    def n_qubits(self) -> int:
        return len(self.codes)

    @property
    def label(self) -> str:
        return self.identifier

    def copy(self) -> PauliString:
        return PauliString(self.identifier, phase=self.phase, ctx=self.ctx)

    def support(self) -> Tuple[int, ...]:
        """
        Return the support of the Pauli operator.

        The support is the tuple of qubit indices on which the operator acts
        nontrivially, i.e. the sites where the local Pauli factor is not the
        identity.

        If

            P = P_0 \\otimes P_1 \\otimes \\cdots \\otimes P_{n-1},

        with each local factor P_k in {I, X, Y, Z}, then

            support(P) = { k : P_k != I }.

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
        """
        Return the Pauli weight of the operator.

        The Pauli weight is the number of qubit sites on which the operator acts
        nontrivially, i.e. the number of local factors that are not the identity.

        If

            P = P_0 \\otimes P_1 \\otimes \\cdots \\otimes P_{n-1},

        with each local factor P_k in {I, X, Y, Z}, then

            weight(P) = |{ k : P_k != I }|.

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
        return all(c == 0 for c in self.codes)

    def adjoint(self) -> PauliString:
        return PauliString(self.identifier, phase=self.phase.conjugate(), ctx=self.ctx)

    dagger = adjoint

    def trace(self):
        if self.is_identity():
            return self.phase * (2 ** self.n_qubits)
        return self.phase * 0.0

    def commutes_with(self, other: PauliString) -> bool:
        self._check_compatible(other)
        anticomm_sites = 0
        for lc, rc in zip(self.codes, other.codes):
            if lc != 0 and rc != 0 and lc != rc:
                anticomm_sites += 1
        return (anticomm_sites % 2) == 0

    def multiply(self, other: PauliString) -> PauliString:
        self._check_compatible(other)
        local_phase, codes = _combine_codes(self.codes, other.codes)
        return PauliString(
            _identifier_from_codes(codes),
            phase=self.phase * other.phase * local_phase,
            ctx=self.ctx,
        )

    def materialize(self) -> DenseArray:
        mats = [self.ctx.ops.asarray(pauli_matrices[ch]) for ch in self.identifier]
        out = reduce(self.ctx.ops.kron, mats)
        return self.phase * out if self.phase != 1 else out

    def matvec(self, x: DenseArray) -> DenseArray:
        x = self.ctx.ops.asarray(x)
        return _apply_codes_to_vector(self.codes, x, self.ctx.ops, phase=self.phase)

    apply = matvec

    def to_sum(self, coeff: complex = 1.0) -> PauliSum:
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
        return self.matvec(other)

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
        return (), (self.identifier, self.phase, self.ctx)

    @classmethod
    def tree_unflatten(cls, aux, children):
        identifier, phase, ctx = aux
        return cls(identifier, phase=phase, ctx=ctx)

    def _convert(self, ctx: Context) -> PauliString:
        return PauliString(self.identifier, phase=self.phase, ctx=ctx)


@jax_pytree_class
@dataclass(init=False)
class PauliSum(ContextBound):
    """Linear combination of Pauli strings with symbolic storage and fast matvec."""

    def __init__(
        self,
        terms: Iterable[str | PauliString],
        coeffs: Iterable[complex] | None = None,
        *,
        ctx: Context | str | None = None,
        simplify: bool = True,
    ) -> None:
        raw_terms = list(terms)
        if not raw_terms:
            raise ValueError("terms must be non-empty")

        term_objs = [
            term if isinstance(term, PauliString) else PauliString(term, ctx=ctx)
            for term in raw_terms
        ]
        resolved_ctx = ctx_manager.resolve_context_priority(ctx, *term_objs)
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
        return self._n_qubits

    @property
    def n_terms(self) -> int:
        return len(self.terms)

    def support(self) -> Tuple[int, ...]:
        """
        Return the support of the Pauli sum.

        The support is the tuple of qubit indices on which at least one Pauli
        term acts nontrivially. Equivalently, it is the union of the supports
        of all Pauli-string terms in the sum.

        If

            A = \\sum_j c_j P_j,

        then

            support(A) = \\bigcup_j support(P_j).

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
        return PauliSum(self.terms, coeffs=self.coeffs, ctx=self.ctx, simplify=True)

    def materialize(self) -> DenseArray:
        out = None
        for coeff, term in zip(self.coeffs, self.terms):
            term_mat = coeff * term.materialize()
            out = term_mat if out is None else out + term_mat
        return out

    def matvec(self, x: DenseArray) -> DenseArray:
        x = self.ctx.ops.asarray(x)
        acc = x * 0
        for coeff, term in zip(self.coeffs, self.terms):
            acc = acc + coeff * _apply_codes_to_vector(term.codes, x, self.ctx.ops, phase=term.phase)
        return acc

    apply = matvec

    def add_term(self, term: str | PauliString, coeff: complex = 1.0) -> "PauliSum":
        if not isinstance(term, PauliString):
            term = PauliString(term, ctx=self.ctx)
        return PauliSum(
            [*self.terms, term],
            coeffs=[*list(self.coeffs), coeff],
            ctx=self.ctx,
            simplify=True,
        )

    def trace(self):
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
        return self.matvec(other)

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
        return (self.coeffs,), (self.terms, self.ctx)

    @classmethod
    def tree_unflatten(cls, aux, children):
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
        """
        Decompose a Hermitian matrix into the Pauli-string basis.

        For a matrix A acting on n qubits, this method computes coefficients
        c_alpha such that

            A = sum_alpha c_alpha P_alpha,

        where the sum runs over all n-qubit Pauli strings and

            c_alpha = 2^{-n} Tr(P_alpha A).

        Args:
            mat:
                Dense matrix of shape (2**n, 2**n).
            tol:
                Terms with |c_alpha| <= tol are discarded.
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
        obj_ctx = ctx_manager.normalize_context(ctx)
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
