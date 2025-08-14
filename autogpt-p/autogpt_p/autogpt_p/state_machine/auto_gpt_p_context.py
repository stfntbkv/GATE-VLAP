from autogpt_p.execution.plan_executor import PlanExecutorInterface
from autogpt_p.state_machine.state import State, Idle


class AutoGPTPContext:

    def __init__(self, llm, max_iterations, memory, executor: PlanExecutorInterface, state=None):
        self.llm = llm
        self.max_iterations = max_iterations
        self.memory = memory
        self.current_iterations = 0
        self.executor = executor

        if state:
            self.state = state
        else:
            self.state = Idle(self)

        self.state.enter()

    def transition(self, state: State):
        self.state.exit()
        self.memory.update_actor_location("robot0")
        self.memory.update_commands()
        self.state = state
        self.state.enter()

    def process_command(self, command):
        self.state.process_command(command, self.memory)
        self.transition(self.state.next())

    def process_failure(self, error):
        self.state.process_failure(error, self.memory)
        self.transition(self.state.next())

    def abort(self):
        self.state.abort()
        self.transition(self.state.next())
