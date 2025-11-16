<div align="center">

# Sentio - team1

[![Pipeline Status][pipeline-badge]][pipeline-url]
[![Coverage Report][coverage-badge]][coverage-url]
[![Latest Release][release-badge]][release-url]

</div>

[pipeline-badge]: https://git.chalmers.se/courses/dit826/2025/team1/badges/main/pipeline.svg
[pipeline-url]: https://git.chalmers.se/courses/dit826/2025/team1/-/pipelines

[coverage-badge]: https://git.chalmers.se/courses/dit826/2025/team1/badges/main/coverage.svg?job=test-with-coverage
[coverage-url]: https://git.chalmers.se/courses/dit826/2025/team1/-/jobs

[release-badge]: https://git.chalmers.se/courses/dit826/2025/team1/-/badges/release.svg
[release-url]: https://git.chalmers.se/courses/dit826/2025/team1/-/releases


## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Environment Settings](#environment-settings)
- [Authors](#authors)
- [Project Planning] (#project-planning)
- [License](#license)


## Description
Sentio is a stress analysis system that examines how interactions and content on social platforms impact users' mental health. Using natural language processing and sentiment analysis, the system evaluates the tone and emotional content of text from Reddit posts to detect stress levels and provide personalized mental health recommendations.

## Installation

### Prerequisits
TBD

### Setup
1. Clone the repository:
```bash
git clone https://git.chalmers.se/courses/dit826/2025/team1.git
cd team1
```

2. Create and activate virtual environment:
```bash
# Using conda/mamba
conda env create -f environment.yml
conda activate ai-project
```

3. Apply database migrations:
```bash
python manage.py migrate
```

4. Create a superuser (for admin access):
```bash
python manage.py createsuperuser
```

5. Run the development server:
```bash
python manage.py runserver
```

The application will be available at `http://127.0.0.1:8000/`

## Usage

### Running Tests
```bash
# Run all tests
python manage.py test

# Run tests for specific app
python manage.py test apps.predictions
```

### Django Shell (for development)
```bash
python manage.py shell
```

## Environment Settings

The project uses split settings for different environments:

- **Development**: `sentio.settings.development` (default)
- **Production**: `sentio.settings.production`
- **Testing**: `sentio.settings.testing`

To use a specific setting:
```bash
DJANGO_SETTINGS_MODULE=sentio.settings.production python manage.py runserver
```

## Authors
- Marcus Berggren
- Lian Shi
- Julia McCall
- Claudia Sevilla Eslava
- Karl Byland

## Project Planning
The project planning markdown file is located at [team1/docs/project_planning.md](docs/project-planning.md).

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

