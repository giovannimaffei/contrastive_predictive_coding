from unittest import TestCase
from cpcspeech.cpcspeech import CPC_speech

class Test_CPC_speech(TestCase):

	def test_output_shape(self):
		cpc_speech_features = CPC_speech()
		input_tensor = torch.random()
		exp_out_size = 
		out = cpc_speech_features.transform(input_tensor)
		self.assertEqual(out.size(),exp_out_size)