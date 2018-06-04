from lalcheck.utils import dataclass


class Task(object):
    def __eq__(self, other):
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError

    def requires(self):
        """
        :rtype: dict[str, Requirement]: The results that this tasks requires,
            as a dictionary from requirement to name.
        """
        raise NotImplementedError

    def provides(self):
        """
        :rtype: dict[str, Requirement]: The results that this tasks provides,
            as a dictionary from name to requirements.
        """
        raise NotImplementedError

    def run(self, **kwargs):
        """
        :param **object kwargs: The actual results required by this task.
        :rtype: dict[str, object]: The actual results provided by this task.
        """
        raise NotImplementedError


class Requirement(object):
    def __eq__(self, other):
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError

    def providers(self):
        """
        Returns the list of task that are able to build this requirement.

        :rtype: list[Task]
        """
        raise NotImplementedError

    @staticmethod
    def as_requirement(provider):
        """
        Creates a requirement from a single function that returns a list of
        tasks. The parameters of the function become attributes of the
        requirement.

        Use as an annotation.

        :param function provider: The function which returns a list of tasks.
        :rtype: Requirement
        """
        def init(self, *args):
            self.args = tuple(args)

        def providers(self):
            return provider(*self.args)

        cls = type(provider.__name__, (Requirement,), {
            '__init__': init,
            'providers': providers
        })

        return dataclass(cls)


class Schedule(object):
    def __init__(self, batches, spec):
        """
        :param list[set[Task]] batches: The list of batches of tasks that
            make the schedule. Each batch is a set of task that can be run
            independently. The ordering of the list indicates the order in
            which the batches will be run.

        :param dict[str, Requirement] spec: The desired results.
        """
        self.batches = batches
        self.spec = spec

    def run(self):
        """
        Runs the schedule.
        :rtype: dict[str, object]
        """
        acc = {}
        for batch in self.batches:
            for task in batch:
                kwargs = {
                    name: acc[req]
                    for name, req in task.requires().iteritems()
                }
                task_res = task.run(**kwargs)
                for name, prov in task.provides().iteritems():
                    acc[prov] = task_res[name]

        return {
            name: acc[req]
            for name, req in self.spec.iteritems()
        }

    def __repr__(self):
        return '\n'.join(repr(b) for b in self.batches)


class Scheduler(object):
    def __init__(self):
        pass

    def schedule(self, spec):
        """
        Given a specification, returns a list of possible schedule that will
        generate the desired results.

        :param dict[str, Requirement] spec: The specification, as a map
            from name to requirement.

        :rtype: list[Schedule]
        """

        # This routine is split in two phases:
        # 1. From the specification "spec", generate the sets of tasks that
        #    will need to be ran (without ordering them) by walking up the
        #    dependency chains. There can be several possibilities due to the
        #    fact that a single requirement may have multiple providers.
        #
        # 2. Order each set of tasks using a simple topological sort and
        #    output it as a new schedule.

        # FIRST PHASE.

        # An option is a pair containing:
        # 1. The set of tasks that need to be run in order to produce all the
        #    requirements seen so far.
        # 2. The set of requirements that are left to consider.
        #
        # At first, there is a single "empty" option (we haven't considered
        # any requirement yet).
        options = [(set(), set(spec.values()))]
        done_options = []

        while len(options) > 0:
            option = options.pop()
            available_tasks = option[0]
            requirements = option[1]

            # Generate the set of tasks for the chosen option. This implies
            # taking care of each requirement, until there are no more.
            while len(requirements) > 0:
                r = requirements.pop()
                providers = r.providers()

                # If there is more than a single way to produce a requirement,
                # branch the current option into new ones, one for each
                # particular way to create the requirement.
                for task in providers[1:]:
                    available_tasks_copy = available_tasks.copy()
                    requirements_copy = requirements.copy()
                    available_tasks_copy.add(task)
                    requirements_copy.update(task.requires().values())
                    options.append((available_tasks_copy, requirements_copy))

                # Add the task that needs to be run in the current option,
                # and remove the requirement that we just dealt with.
                available_tasks.add(providers[0])
                requirements.update(providers[0].requires().values())

            # Once all requirements are taken care of, we have obtained the
            # full set of tasks.
            done_options.append(available_tasks)

        # SECOND PHASE.

        schedules = []

        # Sort each set of tasks independently.
        for available_tasks in done_options:
            # batches will contain subsets of the full set of tasks, such that
            # each task in a subset can be run independently. Moreover, the
            # order in which each subset appears in the list determines the
            # order in which they need to be ran so as to ensure that the
            # dependencies of each task are available before it is run.
            batches = []

            # Contains the tasks that are left to insert in a batch, as a
            # dictionary from task to their "unavailable" dependencies (the
            # requirements that would need to be available before this task
            # can be run).
            task_to_reqs = {
                t: set(t.requires().values())
                for t in available_tasks
            }

            # While there are still tasks that we need to batch...
            while len(task_to_reqs) > 0:
                # Compute the set of tasks which have no dependencies.
                ready = {
                    task
                    for task, reqs in task_to_reqs.iteritems()
                    if len(reqs) == 0
                }

                # If there are none, there must have been a cyclic dependency.
                if len(ready) == 0:
                    raise ValueError("Cyclic dependency found")

                # This set of tasks can be batched right now.
                batches.append(ready)

                # We can now safely remove them from the collection of tasks
                # that are left to batch.
                for task in ready:
                    task_to_reqs.pop(task)

                # Compute the set of requirements that these tasks produce.
                ready_reqs = frozenset(
                    req
                    for task in ready
                    for req in frozenset(task.provides().values())
                )

                # These requirements can be thought as available now that
                # we batched this set of tasks. So, we can remove them from
                # the "unavailable" dependencies of the tasks that are left
                # to batch.
                for reqs in task_to_reqs.itervalues():
                    reqs.difference_update(ready_reqs)

            schedules.append(Schedule(batches, spec))

        return schedules
