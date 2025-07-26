
import sys
import unittest
from dotenv import load_dotenv
import tempfile

load_dotenv()
import os

sys.path.append("m3/demo/experts")

from expert_tb import ExpertTB
from utils import save_image_url_to_file

# TODO: Add a sample image URL for testing
TB_URL = "https://developer.download.nvidia.com/assets/Clara/monai/samples/cxr_00026451_030.jpg"

class TestTBExpert(unittest.TestCase):
    def test_run_tb(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prompt = "This seems a CXR image. Let me trigger <TB>."
            tb_expert = ExpertTB()
            self.assertTrue(tb_expert.mentioned_by(prompt))
            img_file = save_image_url_to_file(TB_URL, temp_dir)
            output_text, _, _ = tb_expert.run(
                image_url=TB_URL,
                input=prompt,
                output_dir=temp_dir,
                img_file=img_file,
                slice_index=0,
                prompt="",
            )

            self.assertTrue(output_text is not None)

if __name__ == "__main__":
    unittest.main()
