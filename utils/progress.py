# Shishir Tandale
class Progress(object):
    #TODO rewrite so you don't use as many floating point ops for speed
    def __init__(self, message, count, precision=2, stride=1):
        self.message = message
        self.size = count
        self.precision = precision
        self.stride = stride/count
    def __enter__(self):
        import sys
        self.progress = 0
        msg_num = lambda prgs, rnd=self.precision: ("\r{}: {:."+str(rnd)+"%}").format(self.message, prgs)
        reset = '\r'+' '*(len(self.message)+7+self.precision)+'\r'
        def p_print():
            msg = msg_num(self.progress) if self.progress < self.size else reset
            sys.stdout.write(f"{msg}")
        def p_update(passthrough=None):
            self.progress += self.stride
            p_print()
            if passthrough is not None:
                return passthrough
        return p_update
    def __exit__(self, exc_type, exc_value, traceback):
        print()

if __name__ == "__main__":
    import numpy as np
    nums = np.random.randint(1, 1_000_000, 10_000_000)
    with Progress("Squarer", len(nums)) as update:
        [update(i**2) for i in nums]
