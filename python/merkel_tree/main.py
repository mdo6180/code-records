from tree import merkle_root, generate_proof, verify_proof

import debugpy
import argparse


argument_parser = argparse.ArgumentParser(description="Simple Merkle Tree Example")
argument_parser.add_argument("-d", "--debug", action="store_true", help="Enable debugging mode.")
args = argument_parser.parse_args()

if args.debug:
    debugpy.listen(5678)
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
    print("Debugger attached")


chunks = [
    b"chunk zero",      # fa7790042e86b05b8b4f479f92e49f74583f4d3169f1e625c1a7838c1710927b
    b"chunk one",       # 9ab05ac9356007b1fddc90108e88bc5e6ec8fdbfad555b797954560de5ef9c62
    b"chunk two",       # ec0af392d2058aec853166cad5c445ee24cc9ff39a8a40cf125308aaf9438a86
    b"chunk three",     # 6c497191b712ad49c83ff8a94b1226e697436c75c6a077c88eea52b5ca20ab2d
]

# left hash:    fba9ba5ececaf82c786209412388661e72b163b17a6ebd1253e8ec027bf09812
# right hash:   f31ba1d6f06af83c5ecf8e8f0d4245c70478d66deaaf3be5b1534f4cbac02222
# root hash:    1d6ae75dbd2d00a91e878275e1daf65d5f1666591212d8dceb2db7f24e8ca58f

#               1d6a...
#           /           \
#    fba9...             f31b...
#   /       \           /       \
# fa77...   9ab0...   ec0a...   6c49...

root = merkle_root(chunks)
print(root.hex())

proof = generate_proof(
    items=chunks,
    leaf_index=2,
)
print(proof[0].side)
print(proof[0].hash.hex())
print(proof[1].side)
print(proof[1].hash.hex())

valid = verify_proof(
    item=chunks[2],
    proof=proof,
    expected_root=root,
)

print(valid)  # True


valid = verify_proof(
    item=b"malicious replacement",
    proof=proof,
    expected_root=root,
)

print(valid)  # False