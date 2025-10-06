import sys
import time

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.primitives import StatevectorEstimator

# Algorithms (0.4.0)
from qiskit_algorithms.minimum_eigensolvers import AdaptVQE, VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SLSQP, SciPyOptimizer

# Nature (0.7.2)
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.units import DistanceUnit

from typing import Dict

CHEM_ACC = 1.6e-3  # Hartree

def _two_qubit_gate_histogram(circuit: QuantumCircuit) -> Dict[str, int]:
    """Return histogram of two-qubit gates by name for the supplied circuit."""
    hist: Dict[str, int] = {}
    for instr in circuit.data:
        if hasattr(instr, "operation"):
            op = instr.operation
            qargs = instr.qubits
        else:  # legacy tuple format
            op, qargs, _ = instr
        if len(qargs) == 2:
            hist[op.name] = hist.get(op.name, 0) + 1
    return hist


def _hist_total(hist: Dict[str, int]) -> int:
    return int(sum(hist.values()))

from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper

def build_problem(molecule: str):

    molecules = {
        "H2":   "H 0.3710 0.0 0.0; H -0.3710 0.0 0.0",  # 4Q
        "H4":   "H 1.11 0.0 0.0; H 0.37 0.0 0.0; H -0.37 0.0 0.0; H -1.11 0.0 0.0", # 8Q: 분자구조 불확실
        "H6":   "H 0.74 0.0 0.0; H 0.37 0.64 0.0; H -0.37 0.64 0.0; H -0.74 0.0 0.0; H -0.37 -0.64 0.0; H 0.37 -0.64 0.0", # 12Q: 분자구조 불확실
        "LiH":  "Li 0.3925 0.0 0.0; H -1.1774 0.0 0.0", # 12Q
        "BeH2": "Be 0.0 0.0 0.0; H 1.3295 0.0 0.0; H -1.3295 0.0 0.0", # 14Q
        "NH3":  "N 0.0 0.0 0.1184; H 0.0 0.9325 -0.2763; H 0.8076 -0.4663 -0.2763; H -0.5076 -0.4663 -0.2763", # 16Q
        "CH4":  "C 0.0 0.0 0.0; H 0.6270 0.6270 0.6270; H -0.6270 -0.6270 0.6270; H -0.6270 0.6270 -0.6270; H 0.6270 -0.6270 -0.6270", # 18Q (좌표 오탈자 정정)
        "LiF":  "Li 0.0 0.0 0.0; F 0.0 0.0 1.5639", # 20Q
        "MgH2": "Mg 0.0 0.0 0.0; H 0.0 0.0 1.1621; H 0.0 0.0 -1.1621", # 22Q
        # "N2H2": "N 0.0 0.6234 -0.1198; N 0.0 -0.6234 -0.1198; H 0.0 1.0082 0.8383; H 0.0 -1.0082 0.8383", # 24Q: triplet 가능하다고 함
        "CH3N": "C 0.0 0.0 0.0; H 0.5369 -0.31 0.0; H 0.0 0.62 0.0; H -1.403 -0.19 0.0; N -0.8661 -0.5 0.0", # 26Q (쉼표 제거)
        "LiCl": "Li 0.0 0.0 0.0; Cl 0.0 0.0 1.1621", # 28Q
        "CO2":  "C 0.0 0.0 0.0; O 0.0 0.0 1.1621; O 0.0 0.0 -1.1621", # 30Q
        "C2H6": "C 0.0 0.0 0.7676070; C 0.0 0.0 -0.7676070; H 0.0 1.0305950 1.1676620; H -0.8925220 -0.5152980 1.1676620; H 0.8925220 -0.5152980 1.1676620; H 0.0 -1.0305950 -1.1676620; H -0.8925220 0.5152980 -1.1676620; H 0.8925220 0.5152980 -1.1676620", # 32Q (탭 제거)
    }

    if molecule not in molecules:
        raise KeyError(f"Unknown molecule key: {molecule}. "
                       f"Provide cache or add geometry to the dict.")

    driver = PySCFDriver(
        atom=molecules[molecule],
        basis="sto3g",
        charge=0,  # 전체 전하
        spin=0,    # n_alpha - n_beta
        unit=DistanceUnit.ANGSTROM,
    )
    problem = driver.run()

    mapper = JordanWignerMapper()
    ferm_op = problem.hamiltonian.second_q_op()
    qubit_h = mapper.map(ferm_op)

    return problem, mapper, qubit_h

def run_adaptvqe(atom: str, max_iter: int, debug: int = 0):

    # 1) Build problem + qubit H
    t0 = time.time()
    problem, mapper, qubit_h = build_problem(atom)
    if debug:
        print(f"(DEBUG)problem:\n{type(problem)}\n{problem}", flush=True)
        print(f"(DEBUG)mapper:\n{type(mapper)}\n{mapper}", flush=True)
        print(f"(DEBUG)qubit_h:\n{type(qubit_h)}\n{qubit_h}", flush=True)
    # EX) H2
    # problem: CachedProblem(num_spatial_orbitals=2, num_particles=(1, 1))
    # mapper: <qiskit_nature.second_q.mappers.jordan_wigner_mapper.JordanWignerMapper object at 0x7fd698f92810>
    # qubit_h: SparsePauliOp(['IIII', 'IIIZ', 'IIZI', 'IZII', 'ZIII', 'IIZZ', 'IZIZ', 'ZIIZ', 'YYYY', 'XXYY', 'YYXX', 'XXXX', 'IZZI', 'ZIZI', 'ZZII'],
    #             coeffs=[-0.81280876+0.j,  0.17110568+0.j, -0.22250985+0.j,  0.17110568+0.j,
    # -0.22250985+0.j,  0.12051037+0.j,  0.16859357+0.j,  0.16584097+0.j,
    # 0.0453306 +0.j,  0.0453306 +0.j,  0.0453306 +0.j,  0.0453306 +0.j,
    # 0.16584097+0.j,  0.17432084+0.j,  0.12051037+0.j])
    # SparsePauliOp(['IIXY', 'IIYX'],
    #             coeffs=[ 0.5+0.j, -0.5+0.j])
    # SparsePauliOp(['XYII', 'YXII'],
    #             coeffs=[ 0.5+0.j, -0.5+0.j])
    # SparsePauliOp(['YYXY', 'XYYY', 'XXXY', 'YXYY', 'XYXX', 'YYYX', 'YXXX', 'XXYX'],
    #             coeffs=[-0.125+0.j, -0.125+0.j, -0.125+0.j,  0.125+0.j, -0.125+0.j,  0.125+0.j,
    # 0.125+0.j,  0.125+0.j])

    # 2) Ground-truth (exact) for chemical-accuracy check
    exact = NumPyMinimumEigensolver().compute_minimum_eigenvalue(qubit_h).eigenvalue.real
    print(f"exact_eigenvalue: {exact}", flush=True)
    time_compute_eigenval = time.time() - t0
    if debug:
        print(f"(DEBUG)time_compute_eigenval: {time_compute_eigenval:.1f} sec", flush=True)
    
    # 3) Ansatz = UCCSD (EvolvedOperatorAnsatz) + HF initial state
    t0 = time.time()
    hf = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
    # EX) H2
    # hf: 
    #      ┌───┐
    # q_0: ┤ X ├
    #      └───┘
    # q_1: ─────
    #      ┌───┐
    # q_2: ┤ X ├
    #      └───┘
    # q_3: ─────

    uccsd = UCCSD(
        problem.num_spatial_orbitals,
        problem.num_particles,
        mapper,
        initial_state=None,      # 여기선 초기상태를 ansatz에 넣지 않음
    )
    pool_ops = list(uccsd.operators)  # AdaptVQE에 넘길 풀
    if debug:
        print(f"(DEBUG)num_pool_ops: {len(pool_ops)}", flush=True)
        for i in range(len(pool_ops)):
            print(f"(DEBUG)pool_ops[{i}]: {pool_ops[i]}", flush=True)
    # EX) H2
    # uccsd: 
    #      ┌─────────────────────────────┐
    # q_0: ┤0                            ├
    #      │                             │
    # q_1: ┤1                            ├
    #      │  EvolvedOps(t[0],t[1],t[2]) │
    # q_2: ┤2                            ├
    #      │                             │
    # q_3: ┤3                            ├
    #      └─────────────────────────────┘
    # pool_ops: 
    # pool_ops[0]: SparsePauliOp(['IIXY', 'IIYX'],
    #             coeffs=[ 0.5+0.j, -0.5+0.j])
    # pool_ops[1]: SparsePauliOp(['XYII', 'YXII'],
    #             coeffs=[ 0.5+0.j, -0.5+0.j])
    # pool_ops[2]: SparsePauliOp(['YYXY', 'XYYY', 'XXXY', 'YXYY', 'XYXX', 'YYYX', 'YXXX', 'XXYX'],
    #             coeffs=[-0.125+0.j, -0.125+0.j, -0.125+0.j,  0.125+0.j, -0.125+0.j,  0.125+0.j,
    # 0.125+0.j,  0.125+0.j])

    # 4) VQE + AdaptVQE (operators 미지정: ansatz가 EvolvedOperatorAnsatz 이므로 OK)
    nq = qubit_h.num_qubits
    empty = QuantumCircuit(nq)  # 파라미터 0개짜리 빈 회로가 "초기 ansatz"

    def make_vqe_callback(adapt_logger):
        def callback(eval_count, params, energy, metadata):
            if not adapt_logger.iter_logs:
                return
            entry = adapt_logger.iter_logs[-1]
            info = {
                "eval": int(eval_count),
                "energy": float(energy),
                "params": [float(x) for x in params],
            }
            entry["vqe_eval_count"] = entry.get("vqe_eval_count", 0) + 1
            entry["vqe_last_eval"] = info
        return callback

    BFGS = SciPyOptimizer(
        method="BFGS",
        options={
            "gtol": 1e-8,     # gradient tolerance (원하시는 값)
            "maxiter": 50,  # 반복 상한
        },
    )
    # StatevectorEstimator(): statevector simnulation 기반, noise-free
    # BackendEstimator(): shot-based 구현
    vqe = VQE(StatevectorEstimator(), empty, BFGS, callback=None)
    vqe.initial_point = None  # 파라미터 없음

    from pysrc.custom_adapt_vqe import LoggingAdaptVQE
    adapt = LoggingAdaptVQE(
        vqe,
        operators=pool_ops,          # 여기서 풀을 직접 전달
        initial_state=hf,            # 초기상태는 HF로 지정
        max_iterations=max_iter,
        gradient_threshold=1e-8,
        eigenvalue_threshold=1e-8,
    )
    vqe.callback = make_vqe_callback(adapt)

    res = adapt.compute_minimum_eigenvalue(qubit_h)
    iteration_summary = []
    eig_history = getattr(res, "eigenvalue_history", [])
    for idx, iter_log in enumerate(adapt.iter_logs):
        gradients = [
            float(entry[0]) for entry in iter_log["gradients"]
        ]  # (gradient, metadata) 구조에서 gradient만 추출
        max_grad = max((abs(g) for g in gradients), default=0.0)
        iteration_summary.append(
            {
                "iter": idx + 1,
                "max_gradient": max_grad,
                "all_gradients": gradients,
                "final_energy": float(eig_history[idx].real) if idx < len(eig_history) else None,
                "vqe_eval_count": iter_log.get("vqe_eval_count", 0),
                "vqe_last_eval": iter_log.get("vqe_last_eval"),
            }
        )

    energy = float(res.eigenvalue.real)
    n_iter = getattr(res, "num_iterations", None)
    time_AdaptVQE = time.time() - t0
    
    print(f"energy: {energy}", flush=True)
    print(f"n_iter: {n_iter}", flush=True)
    print(f"eig_history: \n{eig_history}", flush=True)
    print(f"iteration_summary: \n{iteration_summary}", flush=True)
    
    if debug:
        print(f"(DEBUG)time_AdaptVQE: {time_AdaptVQE:.1f} sec", flush=True)

    # 5) 최종 회로 얻기
    t0 = time.time()
    opt = getattr(res, "optimal_circuit", None)
    if debug:
        print(f"(DEBUG)opt:\n{type(opt)}\n{opt}")

    # 6) Two-qubit gate analysis across multiple transpilation scenarios

    twoq_gate_summary: Dict[str, Dict[int, Dict[str, Dict]]] = {
        "all_to_all": {},
        "heavy_hex": {},
    }

    for opt_level in range(4):
        circ = transpile(opt, optimization_level=opt_level)
        hist = _two_qubit_gate_histogram(circ)
        twoq_gate_summary["all_to_all"][opt_level] = {
            "total": _hist_total(hist),
            "by_gate": hist,
        }

    from qiskit_ibm_runtime.fake_provider import FakeBrisbane

    backend = FakeBrisbane()

    layout_mapping: Dict[int, int] = {}
    for opt_level in range(4):
        circ = transpile(opt, backend=backend, optimization_level=opt_level)
        hist = _two_qubit_gate_histogram(circ)
        twoq_gate_summary["heavy_hex"][opt_level] = {
            "total": _hist_total(hist),
            "by_gate": hist,
        }

        if opt_level == 0:
            final_layout = getattr(circ, "layout", None)
            if final_layout is not None and getattr(final_layout, "initial_layout", None) is not None:
                initial_layout = final_layout.initial_layout
                for phys, virt in initial_layout.get_physical_bits().items():
                    if virt is None:
                        continue
                    reg = getattr(virt, "_register", getattr(virt, "register", None))
                    if reg is not None and getattr(reg, "name", "") == "q":
                        layout_mapping[int(phys)] = int(getattr(virt, "_index", getattr(virt, "index", 0)))

    print("[Two-Qubit Gate Summary]", flush=True)
    for scenario in ("all_to_all", "heavy_hex"):
        for opt_level in range(4):
            entry = twoq_gate_summary.get(scenario, {}).get(opt_level)
            if entry is None:
                continue
            print(
                f"  {scenario} opt_level={opt_level}: "
                f"total={entry['total']}, by_gate={entry['by_gate']}",
                flush=True,
            )

    for opt_level in range(4):
        base = twoq_gate_summary["all_to_all"].get(opt_level)
        target = twoq_gate_summary["heavy_hex"].get(opt_level)
        if base is None or target is None:
            continue
        base_total = base["total"]
        target_total = target["total"]
        if base_total:
            change = 100.0 * (target_total - base_total) / base_total
            print(
                "  change opt_level="
                f"{opt_level}: {base_total} -> {target_total} ({change:+.1f}%)",
                flush=True,
            )
        else:
            print(
                "  change opt_level="
                f"{opt_level}: {base_total} -> {target_total} (+inf%)",
                flush=True,
            )

    time_twoq_analysis = time.time() - t0
    if debug:
        print(f"(DEBUG)time_twoq_analysis: {time_twoq_analysis:.1f} sec", flush=True)
    print(f"layout_mapping: {layout_mapping}", flush=True)


if __name__ == "__main__":
    atom = sys.argv[1]
    debug = int(sys.argv[2])
    
    print(f"atom: {atom}")
    
    run_adaptvqe(atom, 1000, debug)
