# Business Brain: Secondary Steel Plant — Context Questionnaire

> **Purpose:** Collect deep operational and business context from induction furnace-based secondary steel plant owners. This data feeds into Business Brain's RAG pipeline, enabling the AI to detect anomalies, surface leakages, and deliver insights that a plant owner's chartered accountant + plant manager combined would miss.

> **Process:** Scrap Iron → Induction Furnace Melting → Continuous Casting / Ingot Casting → Reheating → Rolling Mill → Finished Product (TMT Bars / Billets / Ingots / Wire Rods)

---

## SECTION 1: COMPANY & PLANT BASICS

### 1.1 Identity
| # | Question | Why It Matters |
|---|----------|---------------|
| 1 | Company name (registered) | Legal entity for compliance tracking |
| 2 | Plant location (city, state, industrial area) | State determines power tariff, pollution norms, tax incentives |
| 3 | Year of establishment | Vintage affects equipment condition, depreciation status, lender perception |
| 4 | Udyam/MSME registration number | Enables MSME payment protection (45-day rule), govt scheme eligibility |
| 5 | BIS license number and products covered | Mandatory for TMT bar sales; missing = illegal sales risk |
| 6 | Annual turnover (approx. Rs crore) | Sizes the business for relevant benchmarks |
| 7 | Number of employees (permanent + contract) | Labor cost structure, compliance burden |
| 8 | Promoter/owner background (1st gen / family business / technical) | Affects decision-making style, risk appetite |

### 1.2 Legal & Banking
| # | Question | Why It Matters |
|---|----------|---------------|
| 9 | Primary banker and type of facility (CC/OD/Term Loan) | Interest cost tracking, facility optimization |
| 10 | Total sanctioned credit limits (CC + OD + LC + BG) | Working capital headroom |
| 11 | Current interest rates on each facility | Benchmark against market; flag if overpaying |
| 12 | Do you have LC/BG facilities? What % margin money? | Trapped capital in margin money |
| 13 | GST registration number | ITC reconciliation tracking |
| 14 | Do you file GSTR-1, 3B, 2B reconciliation regularly? | ITC leakage is 1-3% of turnover if mismanaged |
| 15 | Any ongoing disputes (tax, labor, pollution board, legal)? | Risk flags for the system |

---

## SECTION 2: EQUIPMENT & CAPACITY

### 2.1 Induction Furnace
| # | Question | Why It Matters |
|---|----------|---------------|
| 16 | Number of furnace bodies | Twin body = shared power supply optimization |
| 17 | Capacity per furnace body (tons) | Determines melting rate, power requirements |
| 18 | Power supply rating (kW) per furnace | Energy efficiency benchmarking (kWh/ton) |
| 19 | Transformer capacity (KVA) | Over/under-sizing affects efficiency |
| 20 | Operating frequency (Hz) | Higher frequency = better for smaller furnaces |
| 21 | Furnace make/manufacturer | Quality and spare parts availability |
| 22 | Year of installation / last major overhaul | Equipment age = breakdown risk |
| 23 | Power supply type (IGBT / Thyristor-SCR) | IGBT is 8-12% more energy efficient |
| 24 | Average heats per day | Actual throughput vs rated capacity |
| 25 | Average tap-to-tap time (minutes) | Efficiency benchmark (60-90 min is standard) |
| 26 | Average power consumption per ton (kWh/ton) | **Critical metric.** Best: 500, Average: 625, Poor: 800+ |
| 27 | Do you have a dual-body configuration sharing one power supply? | Utilization optimization potential |

### 2.2 Lining & Refractory
| # | Question | Why It Matters |
|---|----------|---------------|
| 28 | Lining material type (Silica / Magnesia / Neutral NRM) | Affects lining life, slag chemistry, alloy recovery |
| 29 | Ramming mass supplier and grade | Quality consistency tracking |
| 30 | Average lining life (number of heats per campaign) | Good: 80-120 heats. Poor: <60 heats = cost leak |
| 31 | Average refractory consumption (kg/ton of steel) | Benchmark: 3.4-3.6 kg/ton |
| 32 | What causes most lining failures? (slag attack / thermal shock / mechanical damage) | Root cause identification |
| 33 | Sintering process followed? (temperature ramp rate) | Proper sintering extends lining life by 30-50% |

### 2.3 Casting
| # | Question | Why It Matters |
|---|----------|---------------|
| 34 | Casting method: CCM / Ingot Mould / Both | CCM yield: 95-97% vs Ingot: 84-88%. Massive impact. |
| 35 | If CCM: number of strands | Throughput capacity |
| 36 | If CCM: billet section sizes produced (mm) | Product range capability |
| 37 | If CCM: average casting speed (m/min) | Productivity benchmark |
| 38 | If Ingot: mould sizes and average mould life | Consumable cost tracking |
| 39 | Tundish capacity and average tundish life | Consumable cost + quality impact |
| 40 | Ladle capacity and type (lined/unlined) | Metal loss in ladle skulls (1-3% of heat) |

### 2.4 Rolling Mill
| # | Question | Why It Matters |
|---|----------|---------------|
| 41 | Do you have your own rolling mill? | Make vs buy (job work) decision |
| 42 | If yes: mill type (continuous / semi-continuous / cross-country) | Efficiency and product range |
| 43 | Number of stands (roughing + intermediate + finishing) | Capacity and capability |
| 44 | Product sizes produced (diameter range for TMT, sections for structural) | Product mix analysis |
| 45 | Reheating furnace type and fuel | Energy cost component (30-35 litres furnace oil/ton) |
| 46 | Do you have inline rolling (hot charge from CCM)? | Eliminates reheating cost entirely |
| 47 | TMT quenching system (Thermex/Tempcore/other) | Quality capability for Fe500D, Fe550D |
| 48 | Average rolling yield (%) | Benchmark: 93-97%. Below 93% = investigation needed |
| 49 | If no own mill: who does your job work? Rate per ton? | Cost comparison for make vs buy |

### 2.5 Other Equipment
| # | Question | Why It Matters |
|---|----------|---------------|
| 50 | Weighbridge: digital/analog? Integrated with ERP? Photo capture? | Weight fraud prevention |
| 51 | Spectrometer make and model (OES for chemistry analysis) | Quality control capability |
| 52 | Overhead crane capacity (tons) | Material handling bottleneck identification |
| 53 | DG set capacity and fuel type | Backup power cost |
| 54 | Water treatment/cooling system type | Cooling efficiency affects furnace performance |
| 55 | Bag filter / fume extraction system installed? | Pollution compliance |

---

## SECTION 3: RAW MATERIAL & PROCUREMENT

### 3.1 Charge Mix
| # | Question | Why It Matters |
|---|----------|---------------|
| 56 | Primary charge mix ratio (% scrap : % sponge iron/DRI : % pig iron) | Determines energy consumption, yield, and slag volume |
| 57 | Scrap grades used (HMS 1, HMS 2, shredded, turnings, local) | Price and quality benchmarking |
| 58 | Sponge iron source (coal-based DRI / gas-based)? Typical Fe content? | Gangue in DRI = higher slag = higher energy cost |
| 59 | Monthly raw material consumption (tons) | Scale of procurement |
| 60 | Average monthly raw material spend (Rs) | Cost structure baseline |

### 3.2 Scrap Procurement
| # | Question | Why It Matters |
|---|----------|---------------|
| 61 | Number of regular scrap suppliers/dealers | Concentration risk |
| 62 | Top 3 suppliers and their approximate share of your total purchase | Supplier dependency analysis |
| 63 | Do you buy imported scrap? From which origins? | Forex risk, lead time risk, quality variability |
| 64 | Payment terms with scrap dealers (advance / cash / 7-day / 30-day) | Working capital impact |
| 65 | How do you verify scrap quality on arrival? (visual / magnet / weighbridge / sampling) | Fraud vulnerability assessment |
| 66 | Have you faced moisture/weight manipulation issues? How often? | Quantifies procurement leakage |
| 67 | How is scrap priced? (per ton basis, grade-wise, lump sum per truck?) | Pricing transparency assessment |
| 68 | Do you track actual yield per supplier lot? | Links supplier quality to production outcome |
| 69 | Average scrap yard inventory (days of consumption) | >15 days = excess capital locked + oxidation risk |
| 70 | Is scrap stored outdoor/indoor? Covered or open? | Monsoon degradation = 5-8% value loss |

### 3.3 Ferro-Alloys & Consumables
| # | Question | Why It Matters |
|---|----------|---------------|
| 71 | Ferro-alloys used (FeSi, FeMn, SiMn, carbon raiser, aluminum) | Alloy cost tracking per heat |
| 72 | Average consumption per ton of steel for each alloy | Benchmark: FeSi 3-6 kg/ton, FeMn 5-10 kg/ton |
| 73 | Number of ferro-alloy suppliers | Concentration risk |
| 74 | Do you test ferro-alloy chemistry on receipt? | Quality fraud detection (declared vs actual grade) |
| 75 | Who supplies refractory/ramming mass? Payment terms? | Consumable cost chain |
| 76 | Flux consumption (lime, fluorspar) per ton? | Benchmark: <1 kg/ton |

---

## SECTION 4: PRODUCTION & QUALITY

### 4.1 Production Metrics
| # | Question | Why It Matters |
|---|----------|---------------|
| 77 | Average daily production (tons of liquid steel) | Capacity utilization calculation |
| 78 | Average daily finished goods output (tons of TMT/billet/ingot) | Overall yield calculation |
| 79 | Rated capacity vs actual production (monthly) | Utilization: India avg 76%. Below 60% = red flag |
| 80 | Number of working days per month | Actual vs possible operating days |
| 81 | Shift pattern (2-shift / 3-shift / continuous) | Capacity utilization potential |
| 82 | Average number of heats per shift | Throughput tracking |
| 83 | Average melting loss (%) | Good: 1-2%. Poor: >4%. Each 1% = Rs 400-500/ton lost |
| 84 | Average overall yield: scrap input to finished product (%) | CCM route: 84-90%. Ingot route: 74-82% |
| 85 | Do you track per-heat data (charge weight, output weight, power, alloys, time)? | Foundation for per-heat cost analysis |
| 86 | How is per-heat data recorded? (manual register / software / SCADA) | Data quality and automation level |

### 4.2 Quality Control
| # | Question | Why It Matters |
|---|----------|---------------|
| 87 | How often do you take spectro samples per heat? (1/2/3 times) | Min 2 (after meltdown + before tap). <2 = quality risk |
| 88 | What grades do you produce? (Fe415 / Fe500 / Fe500D / Fe550D / structural) | Grade mix affects alloy cost and selling price |
| 89 | What is your typical chemistry target for C, Mn, Si, S, P? | Baseline for variance detection |
| 90 | How often do heats fail chemistry? (% of total heats) | Quality rejection rate |
| 91 | Do you do mechanical testing (tensile, bend, re-bend) in-house or external? | Test turnaround affects dispatch speed |
| 92 | What is your customer rejection/complaint rate (% of dispatches)? | Revenue leakage from quality issues |
| 93 | Do you maintain heat-wise traceability (heat number on each bar/billet)? | BIS requirement + customer trust |

### 4.3 Downtime & Maintenance
| # | Question | Why It Matters |
|---|----------|---------------|
| 94 | Average planned downtime per month (hours) | Scheduled maintenance discipline |
| 95 | Average unplanned breakdown per month (hours) | Every hour = 5-7% daily production lost |
| 96 | Most frequent breakdown causes (furnace / CCM / rolling mill / electrical / water system) | Predictive maintenance targeting |
| 97 | Do you follow a preventive maintenance schedule? | Preventive saves 12-18% over reactive |
| 98 | Annual maintenance spend (Rs) | Benchmark against production volume |
| 99 | Do you track MTBF (mean time between failures) for key equipment? | Equipment health monitoring |
| 100 | Do you have SCADA or any automation system? | Data collection automation potential |

---

## SECTION 5: ENERGY & POWER

| # | Question | Why It Matters |
|---|----------|---------------|
| 101 | Power source: state grid / captive / open access / mix? | Cost optimization potential |
| 102 | Contracted demand (KVA) with DISCOM | Demand charges are fixed cost regardless of usage |
| 103 | Average monthly power bill (Rs) | Single largest cost after raw material |
| 104 | Power tariff rate (Rs/kWh) — energy charge component | Benchmark against state industrial rates |
| 105 | Demand charges (Rs/KVA/month) | Fixed cost component |
| 106 | Power factor maintained (average) | Below 0.90 = penalty. IF naturally gives 0.95+ if capacitors are right |
| 107 | Do you have power factor correction (capacitor banks)? | PF penalty avoidance |
| 108 | Do you utilize time-of-day tariff? (shifting melting to off-peak hours) | Off-peak = 15-25% cheaper in many states |
| 109 | Have you done a formal energy audit? When was the last one? | BEE mandate for designated consumers |
| 110 | Any renewable energy (rooftop solar, open access wind/solar)? | Potential Rs 2-4/kWh savings |
| 111 | Monthly power consumption breakdown (furnace vs rolling mill vs auxiliaries) | Identifies where energy is going |
| 112 | Do you have harmonic filters installed? | Harmonics from IF can cause transformer overheating, penalties |

---

## SECTION 6: SALES, CUSTOMERS & PRICING

### 6.1 Product & Pricing
| # | Question | Why It Matters |
|---|----------|---------------|
| 113 | Products sold (TMT bars / billets / ingots / wire rod / structural) and % share | Product mix profitability analysis |
| 114 | Monthly dispatch quantity (tons) by product | Revenue planning baseline |
| 115 | Current selling prices for each product (Rs/ton, ex-works) | Market positioning assessment |
| 116 | How is your selling price determined? (market-linked / cost-plus / negotiated) | Pricing power assessment |
| 117 | Do you sell under your own brand or as unbranded/white-label? | Brand premium potential |
| 118 | What is the typical price differential between your product and top brands (Tata Tiscon, JSW Neo etc.)? | Market positioning gap |

### 6.2 Customer Base
| # | Question | Why It Matters |
|---|----------|---------------|
| 119 | Total number of active buyers | Customer diversification |
| 120 | Top 5 buyers and their approximate % of total sales | **Concentration risk.** Top 3 > 40% = dangerous |
| 121 | Customer types: dealers / builders / fabricators / government / export | Channel mix analysis |
| 122 | Geographic spread of customers (local / state / interstate / export) | Market reach and logistics cost |
| 123 | Average order size (tons per order) | Operational planning |

### 6.3 Credit & Collections
| # | Question | Why It Matters |
|---|----------|---------------|
| 124 | What % of sales are on credit vs cash/advance? | Working capital exposure |
| 125 | Standard credit period offered (days) | Industry norm: 15-45 days |
| 126 | Actual average collection period (days) | If actual >> standard = collection problem |
| 127 | Do you check buyer creditworthiness before extending credit? How? | CIBIL check, trade references, bank references |
| 128 | Do you set credit limits per customer? | Exposure management |
| 129 | Current total receivables outstanding (Rs) | Working capital trapped |
| 130 | Receivables aging: 0-30 / 30-60 / 60-90 / 90+ days (approx. %) | Aging distribution = collection health |
| 131 | Any bad debts in last 2 years? Amount? | Default history |
| 132 | Do you charge interest on overdue payments? | Revenue recovery mechanism |
| 133 | Do you use PDCs (post-dated cheques)? How many bounce monthly? | Cheque bounce rate = buyer health indicator |
| 134 | Do you use any broker/commission agents? What % commission? | Benchmark: 2-5%. Higher = margin leak |

### 6.4 Dispatch & Logistics
| # | Question | Why It Matters |
|---|----------|---------------|
| 135 | Average dispatch per day (tons) | Throughput matching with production |
| 136 | Who arranges transport — you or buyer? | Cost allocation clarity |
| 137 | Average freight cost per ton to key markets | Logistics cost benchmarking |
| 138 | Have you faced weight disputes at buyer end? How often? | Revenue leakage from weighbridge gaps |
| 139 | E-way bill generation: manual or integrated with billing? | Compliance automation level |
| 140 | Average truck turnaround time at plant (hours) | Logistics efficiency |

---

## SECTION 7: FINANCIAL STRUCTURE & COSTS

### 7.1 Cost Structure
| # | Question | Why It Matters |
|---|----------|---------------|
| 141 | What is your approximate cost per ton of finished product? (total) | Baseline for margin analysis |
| 142 | Approximate cost breakdown (% for: raw material, power, alloys, labor, refractory, overhead) | Identifies largest cost levers |
| 143 | Monthly fixed costs (salaries + demand charges + rent + insurance + EMI + interest) | Break-even calculation |
| 144 | What is your current conversion cost? (total cost minus raw material) | Benchmark: Rs 5,000-8,000/ton for IF route |
| 145 | Do you do per-heat costing? | If no, major blind spot |
| 146 | Do you track cost variance (actual vs standard/budget)? | Variance analysis capability |

### 7.2 Working Capital
| # | Question | Why It Matters |
|---|----------|---------------|
| 147 | Average raw material inventory value (Rs) | Capital locked in scrap yard |
| 148 | Average finished goods inventory value (Rs) | Capital locked in dispatch yard |
| 149 | Average receivables outstanding (Rs) | Capital locked with buyers |
| 150 | Average payables outstanding (Rs) | Supplier financing benefit |
| 151 | Estimated cash conversion cycle (days) | RM days + FG days + receivable days - payable days |
| 152 | Peak working capital requirement month (which month, how much?) | Seasonal planning |
| 153 | CC/OD utilization: typically what % of limit is drawn? | >80% = tight. >90% = stress |

### 7.3 GST & Tax
| # | Question | Why It Matters |
|---|----------|---------------|
| 154 | What % of your scrap purchases are from GST-registered dealers? | Unregistered = no ITC = effective 18% cost increase |
| 155 | Have you ever had ITC blocked/reversed? For what reason? | ITC leakage quantification |
| 156 | Do you reconcile GSTR-2B with purchase register monthly? | Non-reconciliation = ITC at risk |
| 157 | Are you aware of the new GST TDS on metal scrap (Oct 2024 provision)? | Compliance awareness |
| 158 | Any pending GST demands or notices? | Tax risk assessment |
| 159 | Accounting software used (Tally / SAP / custom / manual) | Data integration capability |

---

## SECTION 8: OPERATIONS & PROCESS INTELLIGENCE

### 8.1 Process Benchmarks (fill actual values)

These are the numbers Business Brain will continuously monitor for anomalies.

| Parameter | Your Value | Industry Best | Industry Avg | Red Flag |
|-----------|-----------|---------------|-------------|----------|
| Power consumption (kWh/ton melting) | ___ | 500 | 625 | >750 |
| Melting loss (%) | ___ | 1-2% | 3% | >5% |
| Lining life (heats/campaign) | ___ | 100-120 | 80 | <60 |
| Refractory consumption (kg/ton) | ___ | 3.4 | 4.0 | >5.0 |
| Overall yield scrap→finished (%) | ___ | 90% (CCM) | 85% | <80% |
| Tap-to-tap time (minutes) | ___ | 60 | 80 | >100 |
| Rolling yield (%) | ___ | 97% | 95% | <93% |
| Scale loss in rolling (%) | ___ | 1.5% | 2.0% | >2.5% |
| Crop loss in rolling (%) | ___ | 1.0% | 1.5% | >2.5% |
| Tapping temperature (deg C) | ___ | 1640 | 1660 | >1690 |
| Furnace utilization (%) | ___ | 85% | 76% | <60% |

### 8.2 Slag & Waste
| # | Question | Why It Matters |
|---|----------|---------------|
| 160 | Average slag generation per ton (kg) | Benchmark: 11-15 kg/ton (scrap route), 30-50 kg/ton (DRI heavy) |
| 161 | How is slag disposed/sold? Price per ton? | Revenue from waste stream |
| 162 | Scale (mill scale) collection and sale? | Rs 3,000-5,000/ton revenue potential |
| 163 | Do you measure and record slag weight per heat? | Data for yield optimization |

### 8.3 Water & Cooling
| # | Question | Why It Matters |
|---|----------|---------------|
| 164 | Cooling water system: open loop / closed loop / cooling tower? | Water cost and environmental compliance |
| 165 | Water consumption per ton of steel (litres) | Environmental benchmarking |
| 166 | Any water quality issues (scaling, corrosion in coils)? | Equipment damage risk |

---

## SECTION 9: PEOPLE & ORGANIZATION

| # | Question | Why It Matters |
|---|----------|---------------|
| 167 | Organizational structure (Owner-managed / professional management / family-run) | Decision-making speed and data culture |
| 168 | Key roles: who handles procurement? Sales? Finance? Quality? | Data ownership mapping |
| 169 | Number of permanent staff vs contract labor | Cost structure and compliance burden |
| 170 | Average furnace operator experience (years) | Skill directly affects yield and energy consumption |
| 171 | Do you have a dedicated quality head? | Quality discipline indicator |
| 172 | Staff turnover rate (especially furnace operators and quality staff) | Knowledge loss risk |
| 173 | Any ERP/software used for production tracking? | Data integration potential |
| 174 | What reports does management review daily/weekly/monthly? | Current information flow |

---

## SECTION 10: COMPLIANCE & RISK

| # | Question | Why It Matters |
|---|----------|---------------|
| 175 | State Pollution Control Board: CTO valid till? Last inspection? | Regulatory risk (shutdown orders) |
| 176 | BIS certification: valid till? Last surveillance audit? | Sales legality for TMT bars |
| 177 | Factory license: valid? Last renewal? | Operating legality |
| 178 | Fire NOC status? | Safety compliance |
| 179 | EPF/ESI compliance current? | Labor compliance risk |
| 180 | Any past accidents or safety incidents? Workers' compensation claims? | Insurance and liability risk |
| 181 | Insurance coverage: plant & machinery, stock, fire, workmen compensation? | Risk transfer adequacy |
| 182 | Do you have a formal safety officer? | Factory Act requirement for certain plant sizes |

---

## SECTION 11: MARKET INTELLIGENCE & STRATEGY

| # | Question | Why It Matters |
|---|----------|---------------|
| 183 | Who are your top 3 competitors in the region? | Competitive landscape |
| 184 | What is your perceived competitive advantage? (price / quality / relationships / location) | Strategic positioning |
| 185 | Do you track daily scrap and finished goods market prices? How? | Market intelligence maturity |
| 186 | Which markets/platforms do you monitor? (SteelMenu, BigMint, OfBusiness, dealer calls) | Data source mapping |
| 187 | Seasonal demand pattern you observe (best months, worst months) | Demand cycle for planning |
| 188 | Any expansion plans in next 2 years? (capacity / new products / backward integration) | Strategic direction |
| 189 | Any plans for captive power or solar? | Energy cost reduction strategy |
| 190 | Are you considering value-added products (Fe550D, CRS, structural steel)? | Product evolution path |

---

## SECTION 12: DATA SYSTEMS & CURRENT PAIN POINTS

### 12.1 Current Data Landscape
| # | Question | Why It Matters |
|---|----------|---------------|
| 191 | List all software/systems currently used (Tally, VEGA, SCADA, Excel, custom software, WhatsApp groups) | Data source inventory for Business Brain integration |
| 192 | Which of these can export data (CSV/Excel/API)? | Integration feasibility |
| 193 | What data is currently tracked only in paper registers? | Digitization opportunity |
| 194 | Do you use Google Sheets for any tracking? | Direct integration possible |
| 195 | Who currently prepares MIS reports? How long does it take? | Automation opportunity |

### 12.2 Pain Points & Priorities
| # | Question | Why It Matters |
|---|----------|---------------|
| 196 | What is the single biggest "leak" you suspect but can't quantify? | Top priority for Business Brain |
| 197 | What business question keeps you up at night? | Directs AI analysis priority |
| 198 | If you could have one dashboard showing real-time data, what would it show? | UI/insight prioritization |
| 199 | What decisions do you currently make based on gut feel rather than data? | AI augmentation opportunity |
| 200 | How do you currently detect if something is going wrong (material theft, quality slip, cost overrun)? | Current detection gaps |

---

## LEAKAGE MAP: Where Money Disappears

This is the summary framework Business Brain will use to continuously monitor for leakages:

```
PROCUREMENT LEAKAGES (2-5% of RM cost)
├── Scrap weight manipulation (weighbridge fraud)
├── Scrap grade mixing (HMS1 billed, HMS2 delivered)
├── Moisture content inflation (monsoon: 5-8% value loss)
├── Ferro-alloy under-grading (declared 75% Si, actual 70%)
├── Refractory quality inconsistency (lining life drops)
└── Unregistered supplier = lost GST ITC (18% cost increase)

PRODUCTION LEAKAGES (Rs 500-2,000/ton)
├── Excess melting loss (>2% = Rs 400-500/ton per %)
├── Power overconsumption (each 100 kWh/ton extra = Rs 800-1,000)
├── Excessive superheating (+50°C = 25 kWh/ton waste)
├── Poor lining life (<60 heats = 30-50% higher refractory cost)
├── Alloy recovery losses (additions before slag removal)
├── Ladle/tundish skull losses (1-3% of heat weight)
├── Ingot vs CCM yield gap (12-16% vs 3-5% loss)
├── Rolling scale + crop + cobble losses (3-7%)
└── Unoptimized tap-to-tap time (low furnace utilization)

SALES & REVENUE LEAKAGES (1-3% of revenue)
├── Below-market selling prices (no real-time benchmarking)
├── Excessive broker commissions (2-5% of price)
├── Customer grade rejection/downgrades
├── Dispatch weight disputes (0.5-2% shrinkage)
└── Concentration risk (top 3 buyers > 40%)

FINANCIAL LEAKAGES (Rs 200-1,500/ton)
├── Receivables elongation (each 15 extra days = Rs 100-250/ton interest)
├── Excess inventory carrying cost (>15 days scrap = capital locked)
├── GST ITC blocked/reversed (1-3% of turnover)
├── Power tariff inefficiency (wrong TOD, no PF correction, excess demand charges)
├── Unhedged forex on imported scrap (2% INR move = Rs 650-700/ton)
├── Suboptimal bank facility utilization (margin money, excess limits)
├── CC/OD interest on trapped working capital (10-14% p.a.)
└── No early payment discount capture from suppliers

COMPLIANCE RISK (binary: shutdown or fine)
├── PCB Consent to Operate expiry
├── BIS certification lapse (TMT bars become unsellable)
├── GST non-compliance notices
├── Labor law violations (EPF/ESI arrears)
└── Factory license / fire NOC expiry

STRATEGIC LEAKAGES (long-term margin erosion)
├── Not shifting from ingot to CCM route (yield gap)
├── Not owning rolling mill (paying job work premium)
├── Not adopting renewable energy (Rs 2-4/kWh savings possible)
├── Product mix not optimized for margin/furnace-hour
├── No brand building (perpetual price-taker discount)
└── Customer base not diversified (channel/geography)
```

---

## METRIC THRESHOLDS TO AUTO-SET

Once the questionnaire is filled, Business Brain should auto-create these thresholds:

| Metric | Normal | Warning | Critical |
|--------|--------|---------|----------|
| Power consumption (kWh/ton) | 500-625 | 625-750 | >750 |
| Melting loss (%) | 1-2 | 2-4 | >4 |
| Lining life (heats) | 80-120 | 60-80 | <60 |
| Overall yield (%) | 85-92 | 80-85 | <80 |
| Receivables aging >90 days (% of total) | 0-5 | 5-15 | >15 |
| Customer concentration top 3 (% revenue) | 0-30 | 30-50 | >50 |
| CC/OD utilization (%) | 0-70 | 70-85 | >85 |
| Tap-to-tap time (minutes) | 60-80 | 80-100 | >100 |
| Power factor | 0.95-1.0 | 0.90-0.95 | <0.90 |
| Daily production vs capacity (%) | 75-95 | 60-75 | <60 |
| Scrap inventory (days) | 7-15 | 15-25 | >25 |
| Finished goods inventory (days) | 3-7 | 7-15 | >15 |
| Collection period actual vs standard (days over) | 0-5 | 5-15 | >15 |
| Rolling yield (%) | 95-98 | 93-95 | <93 |
| Refractory consumption (kg/ton) | 3.0-3.6 | 3.6-5.0 | >5.0 |

---

## CONTEXT TEXT GENERATION

When the questionnaire is submitted, Business Brain should auto-generate context text like:

> "{Company} is a secondary steel plant in {Location}, {State}, operating {N} induction furnaces of {X}T capacity each with {Y}kW power supply (total rated capacity: {Z} TPD). They use a {scrap%/DRI%/pig iron%} charge mix and produce {products} via {CCM/ingot} casting route{+ own rolling mill / job work rolling}. Their top 3 customers represent {X}% of revenue with average credit period of {Y} days. Current power consumption is {X} kWh/ton against a benchmark of 500-625. Lining life averages {X} heats. Key pain point: {pain point from Q196}. Current data systems: {list from Q191}."

This context enriches every AI query with plant-specific knowledge.
