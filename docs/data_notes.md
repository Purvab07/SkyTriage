SDR data notes:


Source: FAA Service Difficulty Reports, yearly CSV downloads (2024, 2025, 2026 to Nov 12). Explored: 2026-11-13.

Dimension of data:
185,520 reports in total: 66,080 (2024), 67,652 (2025), 51,788 (2026, partial year).
77 columns, identical across all three files. No duplicate OperatorControlNumber.
About 5,000-6,000 reports per month, steady across the period.
Dates: DifficultyDate is MM/DD/YYYY; SubmissionDate is an ISO timestamp with time zone.

Required columns:
Role: Label = what the model predicts. Feature = a model input. Key = used to look up, link or filter records, never a model input. Context = stored and shown to users or used in retrieval/ranking, not in the main classifier.

Column	Filled	Role	Use:-

JASCCode	100%	Label	4-digit system code; first 2 digits = ATA chapter
Discrepancy	100%	Feature	Free-text description; the main model input
AircraftMake / AircraftModel	~100%	Feature	Aircraft type; also used for effectivity and trends
AircraftTotalTime / Cycles	99% / 98%	Feature	Age and usage; risk scoring
StageOfOperationCode	100%	Feature	Phase of flight when found
HowDiscoveredCode	100%	Feature	How it was found
NatureOfConditionA	100%	Feature (ranking)	Coded nature of the problem
DifficultyDate	100%	Key	When it happened; trends and time-based splits (not a model input)
SubmissionDate	100%	Context	When it was reported; reporting lag
OperatorControlNumber	100%	Key	Primary key; links evidence back to its source report
AircraftSerialNumber	99.8%	Key	AD effectivity: checks whether an AD's serial range covers a plane
RegistryNNumber	99.5%	Key	Tail number: detects the same problem returning on the same plane
SubmitterTypeCode	100%	Key	Filters by kind of submitter / operator segment; checks for bias
PartNumber	46%	Key	AD matching when present
PartName / PartCondition	100%	Context	Structured defect summary for retrieval and display; leaks the label, so not a classifier input
PartLocation	98%	Context	Where on the aircraft
PrecautionaryProcedureA	100%	Context	What the crew/operator did

Model inputs at a glance:

Predict: ATA chapter (first 2 digits of JASCCode).
Main classifier input: Discrepancy only, because that is all a user pastes at triage time.
Risk / ranking inputs: aircraft make/model, total time, total cycles, stage of operation, how discovered, nature of condition.
Never inputs: serial number, tail number, control number, submitter type, dates.

Findings that change the design:

The data is about airliners, not trainers. About 96% of reports are Boeing, Airbus, Bombardier/CRJ, Embraer and similar types; 93% come from submitter type A. Only about 7,500 reports (4%) are other makes, mostly business jets (Citation 680A/560XL), helicopters and turboprops. Typical trainers (172, PA-28, PA-44, SR2x, DA4x) have about 800 reports in three years. Similar-case search and trend detection will be thin for flight schools; AD matching is unaffected.
The fix is usually in the text. About 60% of discrepancies contain repair words (replaced, repaired, R&R, corrective action). There's no separate corrective-action column, so the fix must be extracted from Discrepancy. Good news for "why flagged" precedents.
Heavy class imbalance. ATA 53 (fuselage) is 38% of all reports; 33 (lights), 52 (doors), 57 (wings) and 25 (equipment/furnishings) follow. Report macro-F1 and per-class recall, not accuracy.
Label granularity. 486 distinct JASC codes, 51 ATA chapters. Predict the chapter first; treat the full 4-digit code as a stretch goal.
Text style. All upper case. Median 32 words (mean 42, max 1,463). Full of manual references (AMM, FIM, SRM task numbers), station numbers and part numbers. Normalization should keep part numbers and strip or tag manual references.
Leakage risk. PartName and PartCondition strongly hint at the JASC code. At triage time a user only pastes text, so train the main classifier on Discrepancy alone and keep the structured fields for retrieval and ranking.
Repeat reports. 44% of reports share a tail number and JASC code with another report within 30 days, but heavy maintenance checks file many reports at once. Repeat-based counter-evidence must exclude same-day/same-check bundles before it means "the fix didn't hold".