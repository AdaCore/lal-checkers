
class Capability(object):
    """
    A capability object wraps a predicate on a domain.
    It is used to test whether a domain implements a certain feature.

    For example:
    - The HasSplit capability on Intervals is Capability.Yes because all
      intervals domains implement the "split" method.

    - The HasSplit capability on Product is defined such that it is True if and
      only if all child domains of the product also implement the "split"
      method (see the Capability.IfAll constructor)
    """
    def __init__(self, domain_predicate):
        """
        :param AbstractDomain -> bool domain_predicate: A predicate
        """
        self.domain_predicate = domain_predicate

    def __call__(self, domain):
        return self.domain_predicate(domain)

    @staticmethod
    def _if_single(get_domain, capability):
        """
        Returns the capability that is True if and only if the domain
        retrieved through get_domain has the given capability.

        Note: equivalent to _if_all(lambda d: [get_domain(d)], capability)

        :param AbstractDomain -> AbstractDomain get_domain: The function
            which returns the domain to test.
        :param Capability capability: The capability to test on the retrieved
            domain.
        :rtype: Capability
        """
        return Capability(
            lambda domain: capability(get_domain(domain))
        )

    @staticmethod
    def _if_all(get_domains, capability):
        """
        Returns the capability that is True if and only if all domains
        retrieved through get_domains have the given capability.

        :param AbstractDomain -> AbstractDomain get_domains: The function
            which returns the domains to test.
        :param Capability capability: The capability to test on the retrieved
            domains.
        :rtype: Capability
        """
        return Capability(
            lambda domain: all(
                capability(d) for d in get_domains(domain)
            )
        )

    # Predefine build in capabilities so they are available for completions
    # by IDEs.
    No, Yes, IfSingle, IfAll = None, None, None, None
    HasSplit, HasConcretize = None, None


# Singleton capabilities
Capability.No = Capability(lambda domain: False)
Capability.Yes = Capability(lambda domain: True)


# Capability constructors
Capability.IfSingle = staticmethod(Capability._if_single)
Capability.IfAll = staticmethod(Capability._if_all)

# Dynamic capabilities
Capability.HasSplit = Capability(lambda domain: domain.HasSplit(domain))
Capability.HasConcretize = Capability(
    lambda domain: domain.HasConcretize(domain)
)
