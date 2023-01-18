import unittest
import io

import baseline.utils.dataset.pubtator as P

SAMPLE_PUBTATOR_TEXT = """
28483577|t|Formoterol and fluticasone propionate combination improves histone deacetylation and anti-inflammatory activities in bronchial epithelial cells exposed to cigarette smoke.
28483577|a|BACKGROUND: The addition of long-acting beta2-agonists (LABAs) to corticosteroids improves asthma control. Cigarette smoke exposure, increasing oxidative stress, may negatively affect corticosteroid responses. The anti-inflammatory effects of formoterol (FO) and fluticasone propionate (FP) in human bronchial epithelial cells exposed to cigarette smoke extracts (CSE) are unknown. AIMS: This study explored whether FP, alone and in combination with FO, in human bronchial epithelial cellline (16-HBE) and primary bronchial epithelial cells (NHBE), counteracted some CSE-mediated effects and in particular some of the molecular mechanisms of corticosteroid resistance. METHODS: 16-HBE and NHBE were stimulated with CSE, FP and FO alone or combined. HDAC3 and HDAC2 activity, nuclear translocation of GR and NF-kappaB, pERK1/2/tERK1/2 ratio, IL-8, TNF-alpha, IL-1beta mRNA expression, and mitochondrial ROS were evaluated. Actin reorganization in neutrophils was assessed by fluorescence microscopy using the phalloidin method. RESULTS: In 16-HBE, CSE decreased expression/activity of HDAC3, activity of HDAC2, nuclear translocation of GR and increased nuclear NF-kappaB expression, pERK 1/2/tERK1/2 ratio, and mRNA expression of inflammatory cytokines. In NHBE, CSE increased mRNA expression of inflammatory cytokines and supernatants from CSE exposed NHBE increased actin reorganization in neutrophils. FP combined with FO reverted all these phenomena in CSE stimulated 16-HBE cells as well as in NHBE cells. CONCLUSIONS: The present study provides compelling evidences that FP combined with FO may contribute to revert some processes related to steroid resistance induced by oxidative stress due to cigarette smoke exposure increasing the anti-inflammatory effects of FP.
28483577	0	10	Formoterol	Chemical	MESH:D000068759
28483577	15	37	fluticasone propionate	Chemical	MESH:D000068298
28483577	263	269	asthma	Disease	MESH:D001249
28483577	415	425	formoterol	Chemical	MESH:D000068759
28483577	427	429	FO	Chemical	MESH:D000068759
28483577	435	457	fluticasone propionate	Chemical	MESH:D000068298
28483577	459	461	FP	Chemical	MESH:D000068298
28483577	466	471	human	Species	9606
28483577	629	634	human	Species	9606
28483577	666	672	16-HBE	CellLine	CVCL_0112;NCBITaxID:9606
28483577	850	856	16-HBE	CellLine	CVCL_0112;NCBITaxID:9606
28483577	921	926	HDAC3	Gene	8841
28483577	931	936	HDAC2	Gene	3066
28483577	1013	1017	IL-8	Gene	3576
28483577	1019	1028	TNF-alpha	Gene	7124
28483577	1030	1038	IL-1beta	Gene	3552
28483577	1074	1077	ROS	Chemical	-
28483577	1211	1217	16-HBE	CellLine	CVCL_0112;NCBITaxID:9606
28483577	1256	1261	HDAC3	Gene	8841
28483577	1275	1280	HDAC2	Gene	3066
28483577	1643	1649	16-HBE	CellLine	CVCL_0112;NCBITaxID:9606
28483577	1819	1826	steroid	Chemical	MESH:D013256

28483578|t|MHC II-, but not MHC II+, hepatic Stellate cells contribute to liver fibrosis of mice in infection with Schistosoma japonicum.
28483578|a|Hepatic stellate cells (HSCs) are considered as the main effector cells in vitamin A metabolism and liver fibrosis, as well as in hepatic immune regulation. Recently, researches have revealed that HSCs have plasticity and heterogeneity, which depend on their lobular location and whether liver is normal or injured. This research aimed to explore the biological characteristics and heterogeneity of HSCs in mice with Schistosoma japonicum (S. japonicum) infection, and determine the subpopulation of HSCs in pathogenesis of hepatic fibrosis caused by S. japonicum infection. Results revealed that HSCs significantly increased the expressions of MHC II and fibrogenic genes after S. japonicum infection, and could be classified into MHC II+ HSCs and MHC II- HSCs subsets. Both two HSCs populations suppressed the proliferation of activated CD4+T cells, whereas only MHC II- HSCs displayed a myofibroblast-like phenotype. In response to IFN-gamma, HSCs up-regulated the expressions of MHC II and CIITA, while down-regulated the expression of fibrogenic gene Col1. In addition, praziquantel treatment decreased the expressions of fibrogenic genes in MHC II- HSCs. These results confirmed that HSCs from S. japonicum-infected mice have heterogeneity. The MHC II- alpha-SMA+ HSCs were major subsets of HSCs contributing to liver fibrosis and could be considered as a potential target of praziquantel anti-fibrosis treatment.
28483578	0	6	MHC II	Gene	111364
28483578	17	23	MHC II	Gene	111364
28483578	69	77	fibrosis	Disease	MESH:D005355
28483578	81	85	mice	Species	10090
28483578	89	98	infection	Disease	MESH:D007239
28483578	104	125	Schistosoma japonicum	Species	6182
28483578	202	211	vitamin A	Chemical	MESH:D014801
28483578	227	241	liver fibrosis	Disease	MESH:D008103
28483578	534	538	mice	Species	10090
28483578	544	565	Schistosoma japonicum	Species	6182
28483578	567	579	S. japonicum	Species	6182
28483578	581	590	infection	Disease	MESH:D007239
28483578	659	667	fibrosis	Disease	MESH:D005355
28483578	681	700	japonicum infection	Disease	MESH:D012554
28483578	772	778	MHC II	Gene	111364
28483578	809	828	japonicum infection	Disease	MESH:D012554
28483578	859	865	MHC II	Gene	111364
28483578	876	882	MHC II	Gene	111364
28483578	992	998	MHC II	Gene	111364
28483578	1110	1116	MHC II	Gene	111364
28483578	1202	1214	praziquantel	Chemical	MESH:D011223
28483578	1274	1280	MHC II	Gene	111364
28483578	1330	1348	japonicum-infected	Disease	MESH:D012554
28483578	1349	1353	mice	Species	10090
28483578	1378	1384	MHC II	Gene	111364
28483578	1451	1459	fibrosis	Disease	MESH:D005355
28483578	1509	1521	praziquantel	Chemical	MESH:D011223
28483578	1527	1535	fibrosis	Disease	MESH:D005355
"""

class LoadPubtatorTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pass
    def setUp(self) -> None:
        self.sample_file = io.StringIO(SAMPLE_PUBTATOR_TEXT)
    def tearDown(self) -> None:
        del self.sample_file

    def test_load_without_exploding(self):
        outs = P.load_pubtator_file(self.sample_file, explode_entity=False)
        self.assertEqual(len(outs), 2)

        out_0 = outs[0]
        self.assertEqual(set(out_0.keys()), {"t", "a", "id", "ents"})
        self.assertEqual(len(out_0["ents"]), 22)
        self.assertEqual(len([ent for ent in out_0["ents"] if ent.get("type", None) == "Chemical"]), 8)
        self.assertEqual(len([ent for ent in out_0["ents"] if ent.get("type", None) == "Disease"]), 1)
        self.assertEqual(len([ent for ent in out_0["ents"] if ent.get("type", None) == "Species"]), 2)
        self.assertEqual(len([ent for ent in out_0["ents"] if ent.get("type", None) == "CellLine"]), 4)
        self.assertEqual(len([ent for ent in out_0["ents"] if ent.get("type", None) == "Gene"]), 7)

        out_1 = outs[1]
        self.assertEqual(set(out_1.keys()), {"t", "a", "id", "ents"})
        self.assertEqual(len(out_1["ents"]), 28)

    def test_load_with_exploding(self):
        outs = P.load_pubtator_file(self.sample_file, explode_entity=True, entity_sep=";")
        self.assertEqual(len(outs), 2)

        out_0 = outs[0]
        self.assertEqual(set(out_0.keys()), {"t", "a", "id", "ents"})
        self.assertEqual(len(out_0["ents"]), 26)
        self.assertEqual(len([ent for ent in out_0["ents"] if ent.get("type", None) == "Chemical"]), 8)
        self.assertEqual(len([ent for ent in out_0["ents"] if ent.get("type", None) == "Disease"]), 1)
        self.assertEqual(len([ent for ent in out_0["ents"] if ent.get("type", None) == "Species"]), 2)
        self.assertEqual(len([ent for ent in out_0["ents"] if ent.get("type", None) == "CellLine"]), 8)
        self.assertEqual(len([ent for ent in out_0["ents"] if ent.get("type", None) == "Gene"]), 7)

        out_1 = outs[1]
        self.assertEqual(set(out_1.keys()), {"t", "a", "id", "ents"})
        self.assertEqual(len(out_1["ents"]), 28)

    def test_load_with_exploding_but_wont_exploded(self):
        outs = P.load_pubtator_file(self.sample_file, explode_entity=True, entity_sep="---NEVER-EXPLODED---")
        self.assertEqual(len(outs), 2)

        out_0 = outs[0]
        self.assertEqual(set(out_0.keys()), {"t", "a", "id", "ents"})
        self.assertEqual(len(out_0["ents"]), 22)
        self.assertEqual(len([ent for ent in out_0["ents"] if ent.get("type", None) == "Chemical"]), 8)
        self.assertEqual(len([ent for ent in out_0["ents"] if ent.get("type", None) == "Disease"]), 1)
        self.assertEqual(len([ent for ent in out_0["ents"] if ent.get("type", None) == "Species"]), 2)
        self.assertEqual(len([ent for ent in out_0["ents"] if ent.get("type", None) == "CellLine"]), 4)
        self.assertEqual(len([ent for ent in out_0["ents"] if ent.get("type", None) == "Gene"]), 7)

        out_1 = outs[1]
        self.assertEqual(set(out_1.keys()), {"t", "a", "id", "ents"})
        self.assertEqual(len(out_1["ents"]), 28)



