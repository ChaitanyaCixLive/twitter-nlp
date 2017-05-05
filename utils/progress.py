# Shishir Tandale
class Progress(object):
    def __init__(self, message="Progress", count=100, precision=3):
        self.message = message
        self.size = count
        self.precision = precision
    def __enter__(self):
        import sys
        self.progress = 0
        msg_num = lambda prgs, rnd=self.precision: ('\r{}: {:.'+str(rnd)+'%}').format(self.message, prgs)
        reset = '\r'+' '*(len(self.message)+7+self.precision)+'\r'
        stride_base = 1/self.size
        def p_step(passthrough=None, stride=1):
            self.progress += stride*stride_base
            if passthrough is not None:
                return passthrough
        def p_print(passthrough=None):
            try:
                msg = msg_num(self.progress) if self.progress < self.size else reset
                sys.stdout.write('{}'.format(msg))
            except KeyboardInterrupt:
                print()
                raise
            if passthrough is not None:
                return passthrough
        def p_update(passthrough=None):
            pl = p_print(p_step(passthrough))
            if passthrough is not None:
                return pl
        return p_update, p_step, p_print
    def __exit__(self, exc_type, exc_value, traceback):
        print()

if __name__ == "__main__":
    import numpy as np
    nums = np.random.randint(1, 1_000_000, 10_000_000)
    with Progress("Squarer", len(nums)) as (update, _, _):
        [update(i**2) for i in nums]
