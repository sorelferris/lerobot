from huggingface_hub import HfApi

hub_api = HfApi()
hub_api.create_tag("sorel/so101-record-0121", tag="v3.0", repo_type="dataset")
