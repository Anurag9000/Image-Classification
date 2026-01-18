import sys
import os
import traceback
import unittest

# Add current directory to path so we can import from 'pipeline' and 'utils'
sys.path.append(os.getcwd())

def run_test_module(name, tests):
    print(f"--- Running {name} tests ---")
    try:
        for test_func in tests:
            print(f"  Executing {test_func.__name__}...")
            test_func()
        print(f"--- {name} tests passed! ---\n")
    except Exception:
        print(f"!!! Error in {name} tests !!!")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Adapters
    from tests.test_adapters import test_lora_injection, test_ia3_injection
    run_test_module("Adapters", [test_lora_injection, test_ia3_injection])

    # Augmentations
    from tests.test_augmentations import test_mixup, test_cutmix, test_tokenmix, test_builders
    run_test_module("Augmentations", [test_mixup, test_cutmix, test_tokenmix, test_builders])

    # Losses
    from tests.test_losses import test_adaface, test_curricularface, test_focalloss, test_supconloss, test_evidentialloss
    run_test_module("Losses", [test_adaface, test_curricularface, test_focalloss, test_supconloss, test_evidentialloss])

    # Backbone
    from tests.test_backbone import test_token_learner, test_mixstyle, test_cbam, test_dha, test_hybrid_backbone
    run_test_module("Backbone", [test_token_learner, test_mixstyle, test_cbam, test_dha, test_hybrid_backbone])

    # Optimizers
    from tests.test_optimizers import test_sam_optimizer, test_lookahead_optimizer, test_model_ema, test_gradient_centralization
    run_test_module("Optimizers", [test_sam_optimizer, test_lookahead_optimizer, test_model_ema, test_gradient_centralization])

    # Utils
    from tests.test_utils import test_checkpoint_save_load, test_swa, test_save_snapshot
    run_test_module("Utils", [test_checkpoint_save_load, test_swa, test_save_snapshot])

    # Pipeline Components
    from tests.test_pipeline import TestPipeline
    # unittest.TestCase classes need a loader, simpler to just rely on unittest.main() or wrap it
    # But run_test_module expects functions. TestPipeline is a class.
    # We should probably skip adding it to run_test_module and just run it separately or fix run_tests.py to handle classes.
    # However, existing run_tests.py seems to work on functions.
    # Let's import the test methods from an instance? No.
    # Standard way: use unittest.TextTestRunner.
    
    print("--- Running Pipeline Integration Tests ---")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPipeline)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        print("!!! Pipeline Tests Failed !!!")
        sys.exit(1)
    print("--- Pipeline Tests Passed! ---\n")

    print("All tests passed successfully!")
