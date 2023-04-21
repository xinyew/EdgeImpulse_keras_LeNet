with open('viz', 'r') as f:
    with open('vizz', 'w') as f1:
        s = f.read()
        for line in s.splitlines():
            for ll in line.split(' '):
                if ll:
                    if float(ll) > 0.9:
                        f1.write('1.0000')
                    else:
                        f1.write('0.0000')
                f1.write(" ")
            f1.write("\n")
