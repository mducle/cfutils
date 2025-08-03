"""
System test - running examples under Mantid
"""

import unittest
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'examples'))
import mantid.simpleapi
mantid.simpleapi.config.setLogLevel(2)

class ExamplesTests(unittest.TestCase):

    def test_fitengy1(self):
        import fitengy_example_1
        self.assertAlmostEqual(fitengy_example_1.fitpars['B20'], 0.08086826242083478, places=4)
        self.assertAlmostEqual(fitengy_example_1.fitpars['B40'], -0.0017723258972703584, places=4)
        self.assertAlmostEqual(fitengy_example_1.fitpars['B44'], -0.005450659667133649, places=4)
        self.assertAlmostEqual(fitengy_example_1.fitpars['B60'], -3.8345083931343835e-05, places=4)
        self.assertAlmostEqual(fitengy_example_1.fitpars['B64'], 0.0017662343010934945, places=4)

    def test_fitengy2(self):
        import fitengy_example_2
        self.assertAlmostEqual(fitengy_example_2.fitpars['B20'], 0.43673, places=3)
        self.assertAlmostEqual(fitengy_example_2.fitpars['B22'], -0.554082, places=3)
        self.assertAlmostEqual(fitengy_example_2.fitpars['B40'], -0.00709843, places=3)
        self.assertAlmostEqual(fitengy_example_2.fitpars['B42'], 0.0342519, places=3)
        self.assertAlmostEqual(fitengy_example_2.fitpars['B44'], -0.0627286, places=3)
        self.assertAlmostEqual(fitengy_example_2.fitpars['B60'], -0.000640931, places=3)
        self.assertAlmostEqual(fitengy_example_2.fitpars['B62'], 0.0019763, places=3)
        self.assertAlmostEqual(fitengy_example_2.fitpars['B64'], -0.00154877, places=3)
        self.assertAlmostEqual(fitengy_example_2.fitpars['B66'], 0.00351393, places=3)

    def test_pointcharge(self):
        import pointcharge_example
        for ii, refv in enumerate([0.20465,0.16333,0.33567,-0.16795,-0.19672,-0.29741,-0.28582]):
            self.assertAlmostEqual(pointcharge_example.pc[ii], refv, places=3)
    
    def test_scipyfit1(self):
        import scipyfit_example
        ev1 = scipyfit_example.cfobjs[0][0].getEigenvalues()
        ev2 = scipyfit_example.cfobjs[1][0].getEigenvalues()
        rv1 = [0., 1.15785467, 7.80325241, 11.35923887, 19.97333235, 31.43695445, 31.56724523, 49.01461594, 49.07560563, 52.7812547, 52.78940473, 55.88903298, 56.29763793]
        rv2 = [0., 1.46901327, 12.3474257, 16.33951946, 26.18542373, 31.10593973, 47.83164421, 63.68471527, 64.44931102, 80.48294904, 86.05212165, 88.83657918, 92.48443397]
        for cal, ref in zip(ev1, rv1):
            self.assertAlmostEqual(cal, ref, places=3)
        for cal, ref in zip(ev2, rv2):
            self.assertAlmostEqual(cal, ref, places=3)

    def test_scipyfit2(self):
        import scipyfit_example_2
        bp = scipyfit_example_2.bp.tolist() if hasattr(scipyfit_example_2.bp, 'tolist') else scipyfit_example_2.bp
        for ii, refv in enumerate([-0.03750007522246722, -0.0042083625080344775, 0.06241457675537504, 0.00021259203564253843, -0.002880954004317537]):
            self.assertAlmostEqual(bp[ii], refv, places=3)


if __name__ == "__main__":
    unittest.main()
