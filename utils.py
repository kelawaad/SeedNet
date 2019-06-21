from collections import namedtuple

Transition = namedtuple('Transition', ['state', 
                                        'action',
                                        'reward',
                                        'new_state',
                                        'final'])

                                        