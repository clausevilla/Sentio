# Weekly Project Planning

## Week 1

| Feature Description | Responsible | Delivered | Integrated | Notes |
|---------------------|-------------|-----------|------------|-----------------------------------------------------------|
| Create spreadsheet to track attendance | Marcus | Yes | Yes | |
| Brainstorm ideas | All members | Yes | Yes | **Plan A:** Social media & stress prediction, **Plan B:** Facial expression recognition, **Plan C:** Song lyric recommendation tool |
| Find datasets for ideas | All members | Yes | Yes | |
| Decide on weekly recurring meetings | All members | Yes | Yes | |
| Start with the report | All members | Yes | Yes | |

## Week 2

| Feature Description | Responsible | Delivered | Integrated | Notes |
|---------------------|-------------|-----------|------------|-------|
| Select topic | All members | Yes | Yes | |
| Brainstorm | All members | Yes | Yes | |
| Set up simple ML toy model | Karl Byland | Yes | No | Completed logistic regression model, but not integrated due to changed dataset |
| Design initial database schema for the project (create an initial data file, set up SQLite) | Claudia Sevilla Eslava | Yes | | |
| Initiate Django project structures (set up project skeleton in GitLab, separate admin, user, model, connect APIs, software architecture) | Marcus Berggren | Yes | Yes | |
| Finalize report for Assignment 1 | Julia McCall | Yes | Yes | |
| Set up initial frontend design | Lian Shi | Yes | | |
| Create project markdown file | Julia McCall | Yes | Yes | |

## Week 3
| Feature Description | Responsible | Delivered | Integrated | Notes |
|---------------------|-------------|-----------|------------|-------|
| Set up data cleaning and preprocessing pipeline | Julia McCall | No | No | Data cleaning pipeline mostly completed, due to be delivered and integrated by the end of next week |
| Set up CI pipeline | Claudia Sevilla Eslava | No | No | Due to the change of datasets, I had to spend more time on adapting the existing infrastructure |
| Implement the frontend interface within the Django framework, including all pages using HTML, CSS, and JavaScript | Lian Shi | Yes | Yes | |
| Set up CSV import infrastructure, and edit model to fit the new dataset (one-hot encoding) | Claudia Sevilla Eslava | Yes | Yes | |
| Set up simple ML toy models | Karl Byland, Marcus Berggren | Yes | Yes | Explore Logistic Regression, RNN, BERT, Transformers |

## Week 4
| Feature Description | Responsible | Delivered | Integrated | Notes |
|---------------------|-------------|-----------|------------|-------|
| Set up data cleaning and preprocessing pipelines | Julia McCall | Yes | Yes | |
| Set up CI pipeline | Claudia Sevilla Eslava | Yes | Yes | |
| Connect UI to prediction model and store result in the database | Karl Byland | No | No | Mostly completed, working on old database model, but need to adapt to new database model |
| Set up models and training in ml-pipeline with evaluation | Marcus Berggren | No | No | Mostly finished, working on incremental training of traditional models and neural networks. |
| Add unit tests for data preprocessing pipeline | Julia McCall | No | No | Data pipeline changes took priority, unit tests for this pipeline are moved to next week |
| Add unit tests for data uploading and model evaluation | Claudia Sevilla Eslava | Yes | Yes | |
| Implement user authentication system including registration, login/logout, profile management(password change and account deletion APIs), add related unit tests. | Lian Shi | Yes | Yes | |
| Implement GDPR compliance features for handling sensitive health data | Lian Shi | No | No | Task scheduled mid-week. Planned completion date: next Monday |

## Week 5
| Feature Description | Responsible | Delivered | Integrated | Notes |
|---------------------|-------------|-----------|------------|-------|
| Set up models and training in ml-pipeline with evaluation | Marcus Berggren | Yes | Yes | |
| Implement GDPR compliance:consent management with middleware verification, privacy notices, data export and add related unit tests | Lian Shi | Yes | Yes | |
| Implement admin frontend, UI connect user metrics to admin UI | Lian Shi | Yes | Yes |  |
| Implement recommendations based on predictions, and word count  | Claudia Sevilla Eslava | Yes | Yes |  |
| Connect training, evaluation and inference with UI | Marcus Berggren | No | No | Mostly done but need to connect all possible configurations before training |
| Add unit tests for data preprocessing pipeline | Julia McCall | Yes | No | Completed, but in the same branch as another task which is not completed yet |
| Connect UI to prediction model and store result in the database | Karl Byland | Yes | Yes |  |
| Set up Docker containers | Julia McCall | Yes | Yes |  |
| Set up preprocessing branching depending on the model type used | Julia McCall | Yes | No | Completed, but in the same branch as another task which is not completed yet |
| Implement training parameter config UI and upload/job status polling | Lian Shi | Yes | Yes |  |



