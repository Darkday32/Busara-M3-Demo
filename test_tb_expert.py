import sys
import os

# Add the experts directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "m3", "demo"))

def test_tb_expert_import():
    try:
        print("Testing TB Expert import...")
        from m3.demo.experts.expert_tb import ExpertTB
        print("SUCCESS: TB Expert imported successfully")
        
        # Try to instantiate the expert
        print("Testing TB Expert instantiation...")
        tb_expert = ExpertTB()
        print("SUCCESS: TB Expert instantiated successfully")
        print(f"Expert name: {tb_expert.get_expert_name()}")
        print(f"Labels: {tb_expert.labels}")
        
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_tb_expert_import()
