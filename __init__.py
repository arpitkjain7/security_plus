import yaml

with open("config.yml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    print(config)
