Project workspace for the ARC research program.
<empty-block/>
## Pages
The research plan and implementation/build notes live under this project.
<page url="https://app.notion.com/p/3be53c84b650819ea711c7bc56820c4d">ARC — Project Description</page>
<page url="https://app.notion.com/p/3be53c84b65081198a34cf8711288ba8">ARC — Build & Implementation Plan</page>
<page url="https://app.notion.com/p/3be53c84b650813b9b20d39d9c91ce61">ARC — Research Plan</page>
<empty-block/>
Please keep track of all tasks in the ARC project tasks database.
<database url="https://app.notion.com/p/3be53c84b650813aa34dc5c32d48bafa" inline="true" data-source-url="collection://de653c84-b650-8394-a721-87ed0bed4ae4"></database>
## 📋 Project Decisions
### Base Model
- **Selected**: JetMoE-8B (for initial prototyping and fast iteration)
- **Rationale**: Appropriate scale for development, compatible with free compute tiers (Kaggle, Colab)
- **Future**: Architecture designed to support expansion to DeepSeekMoE-16B and other MoE models
### Repository
- **Location**: nyvo2010/arc (private)
- **Status**: Created, documentation prepared for initial commit
### Documentation
All core Notion docs converted to markdown and ready for repository:
- PROJECT_[DESCRIPTION.md](http://DESCRIPTION.md)
- RESEARCH_[PLAN.md](http://PLAN.md)
- BUILD_[PLAN.md](http://PLAN.md)
- [README.md](http://README.md)
- pyproject.toml
- .gitignore