from prefect import task, flow
from prefect.deployments import run_deployment

@task(name="print")
def status():
    print("I have sucessfully monitored a task")

@flow(name="monitor")
def monitor():
    status()
    
if __name__ == "__main__":
    monitor()




# prefect deployment build src/main.py:create_pytrends_report \
#   -n google-trends-gh-docker \
#   -q test \
#   -sb github/pytrends \
#   -ib docker-container/google-trends \
#   -o prefect-docker-deployment \
#   --apply