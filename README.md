# Numerai

This repository contains:
- numerai models
- numerai compute infrastructure in the form of a `Dockerfile`

## How to add model

Want to add a new model? Here are some guidelines for you.

Aside from the three rules defined below, actually there is nothing you should commit on. That is the beauty of this structure compared to the previous version. The three rules are:
- create a dedicated folder for your package. And of course, leave the rest intact.
- make your code runnable, e.g. ensure you can run something like `python your_model/predict.py`. This part is essential as we aim at automating predictions submissions on a weekly basis.
- add a `requirements.txt` at the root of your folder. it must include all necessary packages to run your code. If this file does not exist, numerai-compute will fail to automate your task.

Where the former package was relying on a common code base, here you are free to go in your own direction. However, should you want to rely on the old code base, it is still available in `./common`.

`clitai` leverages old code base and legacy dataset. On the other hand `megaclitai`, as suggested in the model name, also leverages old code base but uses mega dataset. For more information on how to leverage the old code base, feel free to look into `(mega)clitai/predict.py`. In line documentation should be sufficient.

## How to automate model

You need following credential files:
- `aws.csv` CLI access to AWS.
    - one time download after creating IAM user.
    - Keep original format
- `model.csv` list of model id.
    - Header name,id included
- `numerai.csv` numerai credential file
    - one time download after creating key pair on numerai
    - Header secret_key,public_id included

For the sake of security, we naturally keep these files private.

Of course you need `numerai-cli` installed. As it should be included in your virtual env, run following set of commands

```bash
# login shell
pipenv shell

# install dependencies if not done already
pip install -r requirements
# or
pipenv install -r requirements

# update Dockerfile and entry.sh

# check https://docs.numer.ai/tournament/compute
# config node
# you should be prompted with your model name
numerai node config

# deploy node
numerai node deploy

# test node
# make sure it sends the predictions as expected!
numerai node test
```

## License

This product is licensed under the MIT license.

## Contribute

Want to work on the project? Any kind of contribution is welcome!

Follow these steps:

- Fork the project.
- Create a new branch.
- Make your changes and write tests when practical.
- Commit your changes to the new branch.
- Send a pull request.

## Contact

Questions? Feel free to reach out at numerai_2021@protonmail.com

## Thank you and see you on the moon!