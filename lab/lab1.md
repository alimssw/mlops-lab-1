CCE 5 Grp 7 register here please: https://nonfloating-proceedingly-sabrina.ngrok-free.dev/register/NTQ6YQ

# Lab 1 - git/dvc and data preparation

Through several labs we will be working on the same project taking it from data preparation to model training tracking registration in mlflow, containerization and CI/CD, Kubernetes, to monitoring and live model update.

The first lab will be about creating git and dvc repos. seeing how they work together. And preparing the data files.

> What you need to know:
> - git is for versioning and sharing *code* and *data pointers*
> - dvc is for versioning the *data* itself
> - dvc and git live in the same folder
> - dvc needs git to version the data config file. Developers can use git alone but ML Engineers cannot use dvc alone.

You will need to have a github account and a dagshub account.

## The Data Use Case

Food-11 is a dataset of images labeled by 11 food categories:

0. "Bread"
1. "Dairy product"
2. "Dessert"
3. "Egg"
4. "Fried food"
5. "Meat"
6. "Noodles-Pasta"
7. "Rice"
8. "Seafood"
9. "Soup"
10. "Vegetable-Fruit"

The images are 512x512 pixels and are already split in 3 folders: *training*, *evaluation*, *validation*. The category is in the first part of the file name.

You can download the full raw data from [https://www.kaggle.com/datasets/karakaggle/food11](https://www.kaggle.com/datasets/karakaggle/food11)

## Project Setup

### GitHub Repo creation

- In your github account create a repo named: *mlops-lab-1*
- Keep it empty
- Clone the newly created repo on your laptop

### Install uv

Your project will be using *uv* package manager. Make sure you have it.

```bash
pip install uv 
# or 
pipx install uv
```

### Init the uv project

Initialize the project folder by typing the following.

```bash
uv init
```

> Question 1: Observe the files created, what do you think they contain.

### Setup dvc

In the root folder of your repo type the following.
```bash
dvc init # created the .dvc folder
git add .dvc .dvcignore
git commit -m "Initialize git and dvc"
git push
```

> Question 2: What are the created files. What do you think they are used for? And which ones should be pushed to git?

### Add dagshub as the remote for dvc

In dagshub web interface, open the drop down under the green 'data' button and check the 'Add dvc remote' and 'Setup credentials' sections. Type the following:

```bash
dvc remote add --global origin https://dagshub.com/<username>/<repo-name>.dvc

dvc remote modify origin --global auth basic
dvc remote modify origin --global user <username>
dvc remote modify origin --global password <password>
```

> Question 3: Where are the credentials stored? and what are the options other than --global? Should the credentials be pushed to github? 

Set as default remote and commit the non-secret config

```bash
dvc remote default origin 

git add .dvc/config
git commit -m "Configure DagsHub as dvc remote"
git push
```

### Adding the data

Add your food11 folder under a data folder. You will end up having these folders:
```
./data/food11_raw/training
./data/food11_raw/evaluation
./data/food11_raw/validation 
```

It is highly recommended to stick to the exact naming.

Add you data folder for tracking

```bash
dvc add data
```

> Question 4: Take a look at the .gitignore file. Explain what happened.

> Question 5: Do you see a .dvc file? What does it contain? 

Git commit/push .gitignore and the pointer data.dvc

```bash
git add data.dvc .gitignore
git commit -m "Track data folder with dvc"
git push # code + .dvc pointer files → GitHub
dvc push # actual data → DagsHub
```

> Question 6: You can check your main branch on the github web UI. Is the code there? Is the data there? Do you have any file that points to the data location. And what about dagshub web UI do you see the data? 

> Question 7: In a completely new temporary folder clone your github repo. Do you see the data folder? What dvc command is needed to get the data folder?

### Python script to prepare the image files

Check how ResNet expects the images dataset to be organised. 

Create a file ./src/food11/data.py 
This python script needs to copy food11_raw structure in 2 new folders under ./data:
- ./data/food11_processed 
- ./data/food11_processed_mini 

These datasets have the following different from food11_raw:
1. the images are already shrunk to 128x128
2. the images are arranged in folders that reflect their categories example: data/food11_processed/training/Bread/*
         data/food11_processed/training/Dairy product/*
         data/food11_processed/training/Dessert/*
         data/food11_processed/training/Egg/*
         ...
3. the food11_processed_mini is just the same as food11_processed except with only 100 or less images under each category. This dataset is used for the development so we make sure that our algorithms are correct.

This is how you would run your code in uv. 

```bash
uv run python ./src/food11/data.py 
```

If for example you need to add pillow library:

```bash
uv add pillow
```

Once the script has run, your data folder contains the new processed datasets alongside the raw one. Track the changes with dvc and commit/push the updated pointer, the same way you did for the raw data:

```bash
dvc add data
git add data.dvc
git commit -m "Add food11_processed and food11_processed_mini"
git push
dvc push
```

### Switching to previous commits both code and data

List the previous commits that contain data.dvc modifications. Then checkout the commit on both git and and dvc.

```bash
git log --oneline -- data.dvc # This shows only commits that touched data.dvc
git checkout <old-commit-hash> 
dvc checkout
```

> Question 8: Do you still see the new folders you created? food11_processed and food11_processed_mini?

Then checkout *main* again.

```bash
git checkout main
dvc checkout
```
