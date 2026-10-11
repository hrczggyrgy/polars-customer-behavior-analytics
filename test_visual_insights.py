import importlib.util
import unittest
from datetime import date
from visual_insights_core import share, top_count, month_windows, wilson
HAS_POLARS=importlib.util.find_spec("polars") is not None
if HAS_POLARS:
    import polars as pl
    from visual_insights_core import concentration, cohort_summary, transition_summary

class Definitions(unittest.TestCase):
    def test_definitions(self):
        self.assertIsNone(share(0,0))
        self.assertEqual(top_count(101,.1),11)
        with self.assertRaises(ValueError): top_count(3,0)
        self.assertEqual(month_windows(date(2024,2,1),date(2024,2,29))[0]["partial"],False)
        self.assertEqual(month_windows(date(2024,3,1),date(2024,2,1)),[])
    def test_wilson(self):
        self.assertEqual(wilson(0,0),(None,None))
        for k,n in [(0,1),(1,1),(7,20)]:
            low,high=wilson(k,n)
            self.assertTrue(0<=low<=k/n<=high<=1)
        with self.assertRaises(ValueError): wilson(2,1)

@unittest.skipUnless(HAS_POLARS,"Polars unavailable")
class Aggregates(unittest.TestCase):
    def test_concentration(self):
        r=concentration(pl.DataFrame({"spend_90d":[70.,20.,10.,0.]}).lazy())
        self.assertEqual(r["curve"][-1],(1.,1.))
        self.assertEqual(r["positive_n"],3)
        self.assertAlmostEqual(r["top"][0]["spend_share"],.7)
        self.assertEqual(concentration(pl.DataFrame({"spend_90d":[0.]}).lazy())["curve"],[])
        with self.assertRaises(ValueError): concentration(pl.DataFrame({"spend_90d":[-1.]}).lazy())
    def test_cohorts(self):
        rows=pl.DataFrame({"customer_id":["A","B"],"first_purchase":[date(2024,1,1)]*2}).lazy()
        baskets=pl.DataFrame({"customer_id":["A","A","B","B"],"date":[date(2024,1,1),date(2024,1,31),date(2024,1,1),date(2024,4,1)]}).lazy()
        data=cohort_summary(baskets,rows,date(2024,3,1),True,1)
        self.assertEqual(data.filter(pl.col("horizon")==30)["rate"][0],.5)
        self.assertIsNone(data.filter(pl.col("horizon")==90)["rate"][0])
        self.assertTrue(all(v is None for v in cohort_summary(baskets,rows,date(2024,3,1),False,1)["rate"]))
    def test_transitions(self):
        before=pl.DataFrame({"customer_id":["A","B","C"],"g":["old"]*3}).lazy()
        after=pl.DataFrame({"customer_id":["A","B","D"],"g":["new","old","new"]}).lazy()
        data,counts=transition_summary(before,after,"g")
        self.assertEqual(counts["matched_n"],2)
        self.assertEqual(counts["origin_unmatched"],1)
        self.assertEqual(counts["destination_unmatched"],1)
        self.assertEqual(data["rate"].sum(),1.)

if __name__=="__main__": unittest.main()
