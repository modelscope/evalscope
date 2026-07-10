import os
import tempfile
import unittest

from evalscope.config import TaskConfig
from evalscope.constants import EvalType, JudgeStrategy
from evalscope.run import run_task


@unittest.skipUnless(
    os.getenv('RUN_TOOLATHLON_REMOTE_SMOKE') == '1',
    'Set RUN_TOOLATHLON_REMOTE_SMOKE=1 to run the Toolathlon official-service smoke test.',
)
class TestToolathlonRemoteSmoke(unittest.TestCase):

    def test_toolathlon_private_find_alita_paper(self):
        """Run one Toolathlon private-mode task against the official remote service."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_cfg = TaskConfig(
                model=os.getenv('TOOLATHLON_LOCAL_MODEL', 'local-model'),
                api_url=os.getenv('TOOLATHLON_LOCAL_BASE_URL', 'http://localhost:8000/v1'),
                api_key=os.getenv('TOOLATHLON_LOCAL_API_KEY', 'dummy'),
                eval_type=EvalType.OPENAI_API,
                datasets=['toolathlon'],
                dataset_args={
                    'toolathlon': {
                        'extra_params': {
                            'task_list': ['find-alita-paper'],
                            'workers': 1,
                            'skip_container_restart': True,
                            'timeout_seconds': 3600,
                        }
                    }
                },
                judge_strategy=JudgeStrategy.RULE,
                limit=1,
                eval_batch_size=1,
                work_dir=tmp_dir,
                no_timestamp=True,
            )
            run_task(task_cfg=task_cfg)


if __name__ == '__main__':
    unittest.main()
