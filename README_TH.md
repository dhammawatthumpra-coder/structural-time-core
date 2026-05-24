# StructuralTime-Core v0.1 (MVP)

ไลบรารีภาษา Python ภาษากลางสำหรับการคำนวณและตรวจสอบแบบจำลอง **เวลาเชิงโครงสร้าง (Structural Time)** โดยเชื่อมโยงและบูรณาการสัจพจน์ตรรกะระดับปรัชญาบริสุทธิ์ (Level A Ontology) เข้ากับระบบสมการเชิงพลศาสตร์สารสนเทศ (Level B Dynamics) และการจำแนกสภาวะระบบเชิงพฤติกรรม (Regime Clustering)

โครงการนี้ออกแบบมาให้มีลักษณะโมดูลาร์ (Modular) และเป็นอิสระจากโดเมนศึกษา (Domain Agnostic) เพื่อให้นักวิจัยในสาขาต่าง ๆ (เช่น ปัญญาประดิษฐ์ สังคมศาสตร์ ชีววิทยา ฟิสิกส์) สามารถแปลงข้อมูลดิบของตนเองเพื่อนำมาวิเคราะห์และตรวจจับการเปลี่ยนผ่านสถานะ (Phase Transition) ได้อย่างเป็นระบบ

---

## 1. โครงสร้างแพกเกจหลัก (Core Modules)

ตัวไลบรารีแบ่งส่วนการทำงานออกเป็น 4 โมดูลหลักตามกรอบทฤษฎีหลัก:

*   **`adapters` (Domain Adapter Layer):** ทำหน้าที่แมปสัญญาณดิบเฉพาะของแต่ละศาสตร์ (เช่น ความคล้ายคลึงข้ามเลเยอร์ของ LLM, ข้อมูลประชากรข้ามช่วงวัย) ให้เป็นพารามิเตอร์เวกเตอร์ $K$-state เชิงสารสนเทศ
*   **`ontology` (Ontology Engine - Level A):** จัดการสเปซพ้นเวลา $S$ และเซตความเข้ากันได้ตรรกะ $V$ รวมถึงตัวตรวจสอบ **Asymmetry Conjecture** บนทราเจกทอรีสายพฤติกรรม
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
    LogicalCompatibilityChecker,
    QuarticPotentialSolver,
    GradientFlowIntegrator,
    TheoryGuidedClustering
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

### 3.3 การจัดกลุ่มเรจิมด้วยคุณลักษณะ $\gamma$ (Analytics)
จำแนกสภาวะพฤติกรรมของเอเจนต์ออกเป็นกลุ่มต่าง ๆ โดยลดอัตราการทับซ้อนระหว่าง Frozen และ Decayed:

```python
clustering = TheoryGuidedClustering()

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

---

## 4. การรันการทดสอบ (Running Tests)

สามารถรันชุดการทดสอบ Unit Tests เพื่อตรวจสอบความสมบูรณ์ของสมการและโมดูลทั้งหมดได้ผ่านคำสั่ง:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```
