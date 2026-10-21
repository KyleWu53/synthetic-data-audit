import unittest
import numpy as np
from audit import reference, validate, score, holm

class AuditTests(unittest.TestCase):
    def test_reference_and_validation(self):
        data = reference(1000,42)
        validate(data,1000)
        self.assertAlmostEqual(data.x.corr(data.y),.8,delta=.06)
        with self.assertRaises(ValueError):
            validate(data.iloc[:-1],1000)
        data.loc[0,'x']=np.inf
        with self.assertRaises(ValueError):
            validate(data,1000)

    def test_holm(self):
        np.testing.assert_allclose(holm([.01,.03,.5]),[.03,.06,.5])

    def test_detects_shift_and_broken_correlation(self):
        real=reference(350,1)
        bad=reference(350,2)
        bad['normal']+=4
        bad['y']=-bad['x']
        result=score(real,bad,3)
        self.assertGreater(result['ks']['normal']['statistic'],.8)
        self.assertGreater(result['correlation_rmse'],.4)
        self.assertGreater(result['discriminator_oof_auc'],.95)

if __name__ == '__main__':
    unittest.main()
