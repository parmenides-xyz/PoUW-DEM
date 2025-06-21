
class MockSGXEnclave:
    def execute_task(self, task_type, data, task_id):
        return {
            'proof_hash': f'mock_proof_{task_id}',
            'grid_impact': 1.5,
            'energy_used': 0.3,
            'computation_time': 1.2
        }
