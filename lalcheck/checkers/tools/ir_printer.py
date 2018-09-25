from lalcheck.ai.irs.basic.tools import PrettyPrinter
from lalcheck.ai.utils import dataclass
from lalcheck.checkers.support.checker import Checker, create_provider
from lalcheck.checkers.support.components import IRTrees

from lalcheck.tools.scheduler import Task, Requirement


@Requirement.as_requirement
def PrintIR(provider_config, files):
    return [IRPrinter(provider_config, files)]


@dataclass
class IRPrinter(Task):
    def __init__(self, provider_config, files):
        self.provider_config = provider_config
        self.files = files

    def requires(self):
        return {
            '{}'.format(i): IRTrees(
                self.provider_config,
                f
            )
            for i, f in enumerate(self.files)
        }

    def provides(self):
        return {
            'res': PrintIR(self.provider_config, self.files)
        }

    def run(self, **irs):
        for i, f in enumerate(self.files):
            ir_f = irs[str(i)]
            print("--- IR FOR FILE {} ---\n\n".format(f))
            for j, fun_ir in enumerate(ir_f):
                fun_name = fun_ir.data.fun_id.f_subp_spec.f_subp_name.text
                print("    {}. {}:\n".format(j, fun_name))
                print(PrettyPrinter.pretty_print(fun_ir))
                print("\n")

        return {'res': []}


class IRPrinterChecker(Checker):
    @classmethod
    def name(cls):
        return "ir_printer"

    @classmethod
    def description(cls):
        return "Prints the intermediate representation of subprograms."

    @classmethod
    def kinds(cls):
        return []

    @classmethod
    def create_requirement(cls, provider_config, analysis_files, args):
        return PrintIR(
            create_provider(provider_config),
            analysis_files
        )


checker = IRPrinterChecker


if __name__ == "__main__":
    print("Please run this checker through the run-checkers.py script")
