import subprocess


def main():
    files = {
        'binary_op': 'binary_ops.comp',
        'unary_ops': 'unary_ops.comp'
    }

    defs = {
        'binary_op_add': (files['binary_op'], ['ADD'])
    }

    for d in defs:
        name = d
        file = defs[d][0]
        args = defs[d][1]

        print(f"processing {name} ...")

        cmd = ["glslc", file]

        for arg in args:
            cmd.append("-D" + arg)

        cmd.append("-o")
        cmd.append("-")

        result = subprocess.run(cmd, capture_output=True)

        print(len(result.stdout))


# result = subprocess.run(["whoami"])


# print((result.stdout))


if __name__ == "__main__":
    # execute only if run as a script
    main()
