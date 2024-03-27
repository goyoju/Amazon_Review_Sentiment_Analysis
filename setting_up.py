import subprocess

def run_script(script_name):
    try:
        subprocess.run(['python3', script_name], check=True)
        print(f"{script_name} done")
    except subprocess.CalledProcessError as e:
        print(f"{script_name} error: {e}")

if __name__ == "__main__":
    run_script('python/tfrecord_convert.py') #converting data
    run_script('python/training_setting.py') #setting training sets
    run_script('python/modeling.py') #modeling