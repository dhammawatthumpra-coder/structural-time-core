# StructuralTime-Core v0.1 (MVP)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20371011.svg)](https://doi.org/10.5281/zenodo.20371011)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhammawatthumpra-coder/structural-time-core/blob/main/QuickStart.ipynb)

ไลบรารีภาษา Python ภาษากลางสำหรับการคำนวณและตรวจสอบแบบจำลอง **เวลาเชิงโครงสร้าง (Structural Time)** โดยเชื่อมโยงและบูรณาการสัจพจน์ตรรกะระดับปรัชญาบริสุทธิ์ (Level A Ontology) เข้ากับระบบสมการเชิงพลศาสตร์สารสนเทศ (Level B Dynamics) และการจำแนกสภาวะระบบเชิงพฤติกรรม (Regime Clustering)

โครงการนี้ออกแบบมาให้มีลักษณะโมดูลาร์ (Modular) และเป็นอิสระจากโดเมนศึกษา (Domain Agnostic) เพื่อให้นักวิจัยในสาขาต่าง ๆ (เช่น ปัญญาประดิษฐ์ สังคมศาสตร์ ชีววิทยา ฟิสิกส์) สามารถแปลงข้อมูลดิบของตนเองเพื่อนำมาวิเคราะห์และตรวจจับการเปลี่ยนผ่านสถานะ (Phase Transition) ได้อย่างเป็นระบบ

---

## ข้อพึงระวังเชิงระเบียบวิธีวิจัยและกรอบแนวคิด (Epistemic Disclaimer)

**หมายเหตุสำคัญสำหรับนักวิจัย:**

ไลบรารีนี้รวมระดับการทำงานที่แตกต่างกันสองระดับของ **Structural Time Framework (STF)** เข้าด้วยกัน:
* **Level A (Ontology Engine):** จำลองตรรกะเชิงสเปซที่พ้นเวลาและตรวจสอบความเข้ากันได้เชิงโครงสร้าง
* **Level B (Dynamics Engine):** จำลองระบบพลศาสตร์เชิงตัวเลขต่อเนื่อง (เช่น บ่อศักย์สมการกำลังสี่) และการจัดกลุ่มข้อมูลเชิงประจักษ์

ทั้งสองระดับนี้ทำงานบน **วัตถุทางคณิตศาสตร์ที่แตกต่างกัน** (ตัวดำเนินการ $K$ ในฐานะ stochastic kernel ในระดับ Ontology กับ เวกเตอร์ $K$ ในระดับ Dynamics) แม้จะเป็นเครื่องมือที่เสริมส่งแนวคิดซึ่งกันและกัน แต่มี **ความเป็นอิสระเชิงประจักษ์ต่อกัน (Epistemically Independent)**:
* การรันจำลองพลศาสตร์ใน Dynamics Engine **ไม่ได้เป็นการพิสูจน์หรือรับรองความถูกต้อง** ของข้อสมมติเชิงตรรกะในระดับ Ontology Engine
* ข้อจำกัดในระดับ Ontology Engine **ไม่ได้ช่วยรับรองความถูกต้องทางคณิตศาสตร์** ของการออกแบบเชิงตัวเลขเฉพาะ (เช่น การตั้งสมการศักย์กำลังสี่ หรือการประเมินค่าความหนาแน่นเวลาประจักษ์ $T(K)$) ใน Dynamics Engine
* คลาส `LogicalCompatibilityChecker` ไม่ได้ทำหน้าที่พิสูจน์ข้อสมมติความไม่สมมาตรทางเวลา (*Asymmetry Conjecture*) ในเชิงคณิตศาสตร์ แต่ทำเพียงตรวจสอบว่าทราเจกทอรีเชิงประจักษ์สอดคล้องกับข้อห้ามเชิงตรรกะของสมมติฐานดังกล่าวหรือไม่
* ตัวแปลงข้อมูล `SociologyAdapter` (ซึ่งถูกย้ายไปที่ `examples/sociology_adapter_demo.py`) ทำงานในฐานะตัวอย่างเชิงเปรียบเปรยและแนวคิดเท่านั้น ไม่ควรนำไปใช้ในการพยากรณ์เชิงปริมาณทางสังคมวิทยาโดยปราศจากเมทริกซ์แปลงค่าเฉพาะที่ผ่านการยอมรับในศาสตร์นั้น ๆ

---

## 1. โครงสร้างแพกเกจหลัก (Core Modules)

ตัวไลบรารีแบ่งส่วนการทำงานออกเป็น 4 โมดูลหลักตามกรอบทฤษฎีหลัก:

*   **`adapters` (Domain Adapter Layer):** ทำหน้าที่แมปสัญญาณดิบเฉพาะของแต่ละศาสตร์ (เช่น ความคล้ายคลึงข้ามเลเยอร์ของ LLM, ข้อมูลความก้าวหน้าการฝึกโครงข่ายประสาทเทียม) ให้เป็นพารามิเตอร์เวกเตอร์ $K$-state เชิงสารสนเทศ
*   **`ontology` (Ontology Engine - Level A):** จัดการสเปซพ้นเวลา $S$ และเซตความเข้ากันได้ตรรกะ $V$ รวมถึงตรวจสอบความเข้ากันได้ภายใต้ **Asymmetry Conjecture** บนทราเจกทอรีสายพฤติกรรม
*   **`dynamics` (Dynamics Engine - Level B):** ตัวคำนวณศักย์ Free Energy สมการกำลังสี่ (Quartic Potential), การจำลองวิถีเปลี่ยนรูป $dK/dt$ ด้วย Runge-Kutta 4th Order (RK4) และการคำนวณความหนาแน่นเวลาเชิงประจักษ์ $T(K)$
*   **`analytics` (Regime Clustering & Visualization):** จัดกลุ่มสภาวะระบบ 5 เรจิม (Active, Critical, Turbulent, Decayed, Frozen) โดยใช้สัมประสิทธิ์การสลายตัว $\gamma$ เข้ามาเป็นตัวแปรช่วยแยกแยะระหว่างสภาวะหยุดนิ่งแบบสมบูรณ์ (Frozen) และแบบสลายตัว (Decayed)

---

## 2. การติดตั้งใช้งาน (Quick Start)

คุณสามารถติดตั้งไลบรารีนี้ในเครื่องแบบ Editable mode (แนะนำสำหรับการทดสอบพัฒนา) หรือติดตั้งโดยตรงผ่าน pip:

```bash
# โคลนโปรเจกต์และติดตั้งในเครื่อง
git clone https://github.com/dhammawatthumpra-coder/structural-time-core.git
cd structural-time-core
pip install -e .
```

หรือสามารถติดตั้งโดยตรงจาก GitHub:
```bash
pip install git+https://github.com/dhammawatthumpra-coder/structural-time-core.git
```

เมื่อติดตั้งเสร็จแล้ว นำเข้าใช้งานผ่าน Python:

```python
import numpy as np
from structural_time_core import (
    TransformerAdapter,
    NeuralNetworkTelemetryAdapter,
    LogicalCompatibilityChecker,
    QuarticPotentialSolver,
    GradientFlowIntegrator,
    HybridRegimeClustering
)
```

---

## 3. ตัวอย่างการประยุกต์ใช้งาน (Usage Examples)

### 3.1 การตรวจสอบตรรกะตาม Asymmetry Conjecture (Level A)
การคัดกรองความไม่สมมาตรเชิงระบบเพื่อป้องกันสภาวะขัดแย้งเชิงตรรกะในระบบที่ซับซ้อน:

```python
checker = LogicalCompatibilityChecker(complexity_threshold=0.6)

# ทราเจกทอรีพฤติกรรมที่สมมาตร/หยุดนิ่ง (Flat Trajectory)
flat_K = np.array([0.5, 0.5, 0.5, 0.5])

# จะถูกปฏิเสธภายใต้สภาพแวดล้อมระบบที่มีความซับซ้อนสูง (0.8 > 0.6)
is_valid = checker.is_compatible(flat_K, system_complexity=0.8)
print("Is compatible:", is_valid)  # แสดงผล: False (Logically Excluded)
```

### 3.2 การจำลองพลศาสตร์เวลาเชิงโครงสร้าง (Level B)
การประมวลผลการเปลี่ยนผ่านเฟสของเวกเตอร์พฤติกรรมผ่านบ่อศักย์กำลังสี่ (Double-well potential) ร่วมกับการคำนวณ $T(K)$:

```python
# F(K) = K^4 - 2*K^2 (มีจุดสมดุลเสถียรที่ -1 และ 1, จุดไม่เสถียรที่ 0)
potential = QuarticPotentialSolver(a=1.0, b=0.0, c=-2.0, d=0.0)
integrator = GradientFlowIntegrator(gamma=0.1, dt=0.01)

# เริ่มจำลองการเคลื่อนที่จาก K = 0.5
K = 0.5
for step in range(100):
    dF = potential.compute_dF_dK(K)
    # ใช้ RK4 step เพื่อเสถียรภาพตัวเลขสูงสุด ป้องกัน gradient explosion
    K = integrator.step_rk4(K, potential.compute_dF_dK, u=0.0)
    print(f"Step {step}: K = {K:.4f}")
```

### 3.3 การจัดกลุ่มเรจิมแบบไฮบริด (Hybrid Regime Clustering)
จำแนกสภาวะพฤติกรรมของเอเจนต์ออกเป็นกลุ่มต่าง ๆ โดยลดอัตราการทับซ้อนระหว่าง Frozen และ Decayed:

```python
clustering = HybridRegimeClustering()

# ข้อมูลการสังเกตการณ์: [E_K, dK_dt, gamma]
# 1. พลังงานและพลศาสตร์ต่ำ แต่ค่าสลายตัวสูญเสียสูง (gamma > 0.4) -> Decayed
# 2. พลังงานและพลศาสตร์ต่ำ และสภาวะเสถียร (gamma <= 0.4) -> Frozen
trajectory_data = np.array([
    [0.2, 0.05, 0.75],  # Sample 1
    [0.1, 0.02, 0.12],  # Sample 2
])

regimes = clustering.fit_predict_regimes(trajectory_data)
print("Regimes mapped:", regimes) 
# แสดงผล: ['Decayed', 'Frozen']
```
```

### 3.4 การประยุกต์ใช้กับข้อมูล Telemetry ของ Deep Learning
การแปลงข้อมูลตัวชี้วัดการเรียนรู้ของโครงข่ายประสาทเทียม (Loss, Accuracy, Weight/Gradient norms) เข้าสู่พารามิเตอร์ $K$:

```python
from structural_time_core import NeuralNetworkTelemetryAdapter

# กำหนดค่าขีดจำกัดสูงสุดของ Weight norm และ Gradient norm เพื่อใช้สเกลค่า
adapter = NeuralNetworkTelemetryAdapter(max_weight_norm=100.0, max_grad_norm=10.0)

raw_telemetry = {
    'train_loss': 0.05,
    'val_loss': 1.62,
    'val_accuracy': 0.35,
    'weight_norm': 74.2,
    'gradient_norm': 0.12
}

K = adapter.map_to_K(raw_telemetry)
print("Complexity, Stability, Error Rate:", K)
# แสดงผล: [0.742, 0.35, 0.618]
```

---

## 4. การรันโปรแกรมตัวอย่างและการสร้างกราฟ (Demos & Visualization)

แพกเกจมาพร้อมกับโปรแกรมตัวอย่างที่รันระบบจำลองและสร้างกราฟรายงานผลอัตโนมัติ:

```bash
# 1. รันการจำลองระดับพื้นฐาน (ศักย์กำลังสี่และวิถีทราเจกทอรี)
python examples/simulation_demo.py

# 2. รันการวิเคราะห์ข้อมูลจำลอง Deep Learning (Grokking และ Mode Collapse)
python examples/nn_telemetry_demo.py
```

กราฟผลลัพธ์จะถูกเซฟในโฟลเดอร์ `examples/` (ประกอบด้วย `potential_well.png`, `trajectory.png`, `regime_clustering.png`, `nn_grokking.png`, `nn_mode_collapse.png` และ `nn_clustering_3d.png`)

---

## 5. การรันการทดสอบ (Running Tests)

สามารถรันชุดการทดสอบ Unit Tests เพื่อตรวจสอบความสมบูรณ์ของสมการและโมดูลทั้งหมดได้ผ่านคำสั่ง:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 6. การอ้างอิงเชิงวิชาการ (Citation & Archiving)

หากคุณใช้โปรเจกต์นี้ในงานวิจัยเชิงวิชาการ โปรดอ้างอิงตามรูปแบบที่กำหนดไว้ในไฟล์ `CITATION.cff`:

*   **Zenodo & DOI:** แนะนำให้นักวิจัยเชื่อมโยง Repository นี้เข้ากับ **Zenodo** เพื่อสร้างรหัส **DOI (Digital Object Identifier)** ประจำตัวเวอร์ชันของไลบรารีอย่างเป็นทางการ สำหรับใช้ในการเขียนเอกสารอ้างอิง (Citations) ในงานวิจัยระดับสากล
