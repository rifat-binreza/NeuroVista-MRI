"""Meaningful edge-case checks; no claim of trained-model validation."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from neurovista.metrics import binary_overlap, classification_report
from neurovista.experiments import experiment_plan, balanced_class_weights
from neurovista.lora import validate_adapter_manifest, generation_deficit
from neurovista.preprocessing import prepare_image
from neurovista.inference import classify

class ReferenceTests(unittest.TestCase):
    def test_overlap_known_counts(self):
        self.assertEqual(binary_overlap([1,1,0,0],[1,0,1,0]), {"dice":.5,"iou":1/3})
        self.assertEqual(binary_overlap([0],[0]), {"dice":1.,"iou":1.})
        self.assertEqual(binary_overlap([1],[0]), {"dice":0.,"iou":0.})
    def test_overlap_rejects_invalid(self):
        for a,b in [([1],[float('nan')]),([1],[2]),([1,0],[1]),([],[])]:
            with self.assertRaises(ValueError): binary_overlap(a,b)
    def test_classification_orientation_and_absent_class(self):
        report=classification_report([0,0,1],[0,1,1],['a','b','c'])
        self.assertEqual(report['confusion_matrix'],[[1,1,0],[0,1,0],[0,0,0]])
        self.assertAlmostEqual(report['accuracy'],2/3)
        self.assertEqual(report['per_class'][2]['f1'],0.)
    def test_plan_21_conditions(self):
        rows=list(experiment_plan()); self.assertEqual(len(rows),21)
        self.assertEqual(len({r.name for r in rows}),21)
        self.assertEqual(sum(sum(r.synthetic_counts.values()) for r in rows),1000)
        for row in rows:
            if row.method=='lora': self.assertEqual(sum(row.real_counts.values())+sum(row.synthetic_counts.values()),2000)
    def test_weighting(self):
        w=balanced_class_weights({'a':250,'b':500}); self.assertEqual(w['a'],2*w['b'])
        with self.assertRaises(ValueError): balanced_class_weights({'a':0})
    def test_lora_missing_metadata(self):
        with self.assertRaises(ValueError): validate_adapter_manifest({'rank':4})
        self.assertEqual(generation_deficit(250),250)
        self.assertEqual(generation_deficit(600),0)
    def test_real_image_preprocessing_and_inference_contract(self):
        class Dummy:
            def predict(self,x,verbose=0):
                assert x.shape==(1,299,299,3)
                return np.array([[.1,.2,.3,.4]])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'fixture.png'; Image.new('L',(9,17),255).save(path)
            array=prepare_image(path,'segmentation')
            self.assertEqual(array.shape,(1,256,256,3)); self.assertTrue(np.all(array==1))
            result=classify(path,Dummy(),labels=['a','b','c','d']); self.assertEqual(result['d'],.4)

if __name__=='__main__': unittest.main()
