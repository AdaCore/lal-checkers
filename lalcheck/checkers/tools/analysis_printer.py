from lalcheck.ai.utils import dataclass
from lalcheck.checkers.support.checker import (
    AbstractSemanticsChecker, create_best_provider
)
from lalcheck.checkers.support.components import (
    AbstractSemantics, ModelConfig
)

from lalcheck.tools.scheduler import Task, Requirement
from lalcheck.tools.logger import log

import re
import os
import errno


@Requirement.as_requirement
def PrintAnalysis(provider_config, model_config, filenames, file_matcher,
                  subp_matcher, file_format):
    return [AnalysisPrinter(provider_config, model_config, filenames,
                            file_matcher, subp_matcher, file_format)]


@dataclass
class AnalysisPrinter(Task):
    def __init__(self, provider_config, model_config, filenames,
                 file_matcher, subp_matcher, file_format):
        self.provider_config = provider_config
        self.model_config = model_config
        self.filenames = filenames
        self.file_matcher = file_matcher
        self.subp_matcher = subp_matcher
        self.file_format = file_format
        self._c_file_matcher = re.compile(file_matcher)
        self._c_subp_matcher = re.compile(subp_matcher)

    def requires(self):
        return {
            '{}'.format(i): AbstractSemantics(
                self.provider_config,
                self.model_config,
                self.filenames,
                f
            )
            for i, f in enumerate(self.filenames)
            if self._c_file_matcher.match(f)
        }

    def provides(self):
        return {
            'res': PrintAnalysis(
                self.provider_config,
                self.model_config,
                self.filenames,
                self.file_matcher,
                self.subp_matcher,
                self.file_format
            )
        }

    @staticmethod
    def _ensure_dir(filename):
        """
        Ensures that the directories along the file path exist, creating them
        along the way when they don't.

        :param str filename: path of the file.
        """
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            try:
                os.makedirs(dirname)
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    log(
                        'error',
                        'Could not create directories along {}'.format(
                            dirname
                        )
                    )
                    raise
        elif os.path.isfile(dirname):
            log('error', 'Path {} is already the path of a file'.format(
                dirname
            ))

    def _actual_filename(self, file_name, subp_name):
        """
        Returns the actual path to use. The actual path name is the
        substitution of the user-provided file format by the filename and the
        subprogram name. If this file already exists, an index is appended
        at the end of the filename until a valid path is produced, or a
        maximal number of attempt has been made.

        :param str file_name: The file that is being processed.
        :param str subp_name: The name of the subprogram being analyzed.
        :rtype: str
        """
        subp_name_attempt = subp_name
        i = 0
        while i < 50:
            actual = self.file_format.format(file_name, subp_name_attempt)

            if not os.path.exists(actual):
                return actual

            i += 1
            subp_name_attempt = "{}-{}".format(subp_name, i)

        raise ValueError

    def run(self, **sem):
        for i, f in enumerate(self.filenames):
            if self._c_file_matcher.match(f):
                sem_f = sem[str(i)]
                for sem_subp in sem_f:
                    subp = sem_subp.orig_subp
                    subp_name = subp.f_subp_spec.f_subp_name.text
                    if self._c_subp_matcher.match(subp_name):
                        out_file = self._actual_filename(f, subp_name)
                        self._ensure_dir(out_file)
                        sem_subp.save_results_to_file(out_file)
                        log('info', 'Saved analysis results to {}'.format(
                            out_file
                        ))

        return {'res': []}


class AnalysisPrinterChecker(AbstractSemanticsChecker):
    @classmethod
    def name(cls):
        return "ir_printer"

    @classmethod
    def description(cls):
        return ("Prints the results of the analysis of each subprogram as a "
                "dot file. Each node in a graph represents a program point, "
                "and an edge between two nodes 'a' and 'b' means that control "
                "can flow between 'a' and 'b'. Additionally, the table under "
                "each node contains the knowledge at this program point of "
                "the values that all variables in the subprogram can take.")

    @classmethod
    def kinds(cls):
        return []

    @classmethod
    def create_requirement(cls, project_file, scenario_vars, filenames, args):
        arg_values = cls.get_arg_parser().parse_args(args)

        return PrintAnalysis(
            create_best_provider(project_file, scenario_vars, filenames),
            ModelConfig(arg_values.typer,
                        arg_values.type_interpreter,
                        arg_values.call_strategy,
                        arg_values.merge_predicate),
            tuple(filenames),
            arg_values.file_matcher,
            arg_values.subp_matcher,
            arg_values.file_format
        )

    @classmethod
    def get_arg_parser(cls):
        parser = AbstractSemanticsChecker.get_arg_parser()
        parser.add_argument('--file-matcher', type=str, default='.*',
                            help="Regex used to filter files for which to "
                                 "print results of the analysis.")
        parser.add_argument('--subp-matcher', type=str, default='.*',
                            help="Regex used to filter subprogram names for"
                                 "which to print results of the analysis.")
        parser.add_argument('--file-format', type=str, default='{}/{}.dot',
                            help="The format string to use. Once the analysis "
                                 "of a subprogram is done, the first {} wil l"
                                 "be replaced by the filename in which this "
                                 "subprogram is defined, and the second one "
                                 "will be replaced by the name of the "
                                 "subprogram. The final string will be used "
                                 "as the path of the output dot file. When "
                                 "multiple subprograms in the same file have "
                                 "the same name, an index is appended at the "
                                 "end of the output file.")
        return parser


checker = AnalysisPrinterChecker


if __name__ == "__main__":
    print("Please run this checker through the run-checkers.py script")
