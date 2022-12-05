import argparse 

# from TNPandMS_lib.optimizationProgram import (
#     accelGradient, 
#     MSA
# )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', choices=['MSA', 'accelGradient'], default='MSA')
    args = parser.parse_args()

    if args.name == 'MSA':
        print('MSA')
    elif args.name == 'accelGradient':
        print('accelGradient')

if __name__ == '__main__':
    main()