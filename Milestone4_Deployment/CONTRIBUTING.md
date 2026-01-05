# Contributing to Supply Chain Delivery Prediction

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Be respectful, inclusive, and professional. All contributors are expected to follow our code of conduct.

## Getting Started

### 1. Fork and Clone
```bash
# Fork repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/supply-chain-delivery-prediction.git
cd supply-chain-delivery-prediction

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/supply-chain-delivery-prediction.git
```

### 2. Set Up Development Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Additional dev tools
```

### 3. Create Feature Branch
```bash
# Update main
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
```

## Development Workflow

### Code Style
We follow PEP 8 with these tools:
- **Black** for code formatting
- **Flake8** for linting
- **isort** for import sorting

```bash
# Format code
black .

# Check linting
flake8 .

# Sort imports
isort .
```

### Writing Tests
All code changes require tests:

```python
# tests/test_preprocessing.py
import pytest
from Milestone2_Preprocessing.milestone2_preprocessing import DataPreprocessingPipeline

def test_missing_values_handling():
    """Test that missing values are handled correctly"""
    pipeline = DataPreprocessingPipeline()
    # Test implementation
    assert result == expected

def test_feature_engineering():
    """Test feature engineering creates correct features"""
    # Test implementation
    assert 'supplier_reliability_score' in features
```

Run tests:
```bash
pytest tests/
pytest tests/test_preprocessing.py -v  # Verbose
pytest --cov=.  # With coverage
```

### Documentation
Update documentation for:
- New features
- API changes
- Configuration options
- Installation steps

### Commit Messages
Follow conventional commits:
```
type(scope): subject

body

footer
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style
- `refactor`: Code refactoring
- `test`: Tests
- `chore`: Dependencies, build

**Examples**:
```
feat(models): add LGBM classifier
fix(preprocessing): handle null values in distance column
docs(readme): update installation instructions
test(app): add prediction endpoint tests
```

### Pull Request Process

1. **Before Creating PR**
   - Update main: `git fetch upstream && git rebase upstream/main`
   - Run tests: `pytest tests/`
   - Check code style: `black . && flake8 .`
   - Update documentation

2. **Create Pull Request**
   - Push to your fork: `git push origin feature/your-feature`
   - Go to GitHub and create PR
   - Fill PR template completely
   - Link related issues

3. **PR Title Format**
   ```
   [Type] Brief description
   [feat] Add LGBM model
   [fix] Fix preprocessing bug
   [docs] Update README
   ```

4. **PR Description**
   ```markdown
   ## Description
   Clear description of changes

   ## Related Issues
   Closes #123

   ## Type of Change
   - [x] New feature
   - [ ] Bug fix
   - [ ] Documentation

   ## Testing
   - [x] Unit tests
   - [x] Integration tests
   - [x] Manual testing

   ## Checklist
   - [x] Code follows style guidelines
   - [x] Tests added/updated
   - [x] Documentation updated
   - [x] No breaking changes
   ```

5. **Review Process**
   - Maintainers review code
   - Address feedback
   - Push updates to same branch
   - Get approval

6. **Merge**
   - Squash commits if needed
   - Delete branch
   - Done!

## Issue Types

### Bug Report
```markdown
## Description
Clear description of bug

## Steps to Reproduce
1. Step 1
2. Step 2

## Expected Behavior
What should happen

## Actual Behavior
What actually happened

## Environment
- Python: 3.10
- OS: Windows
- Branch: main
```

### Feature Request
```markdown
## Description
Feature idea

## Motivation
Why is this needed?

## Implementation
Suggested approach

## Alternatives
Other approaches
```

### Documentation Issue
```markdown
## Description
Missing or unclear documentation

## Location
Which file/section

## Suggestion
Proposed improvement
```

## Project Structure Reference

```
supply-chain-delivery-prediction/
├── Milestone1_EDA/              # Exploratory data analysis
│   └── Milestone1_EDA.ipynb
├── Milestone2_Preprocessing/    # Data preprocessing pipeline
│   ├── milestone2_preprocessing.py
│   ├── run_pipeline.py
│   └── outputs/
├── Milestone3_ModelBuilding/    # Model training & evaluation
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── outputs/
├── Milestone4_Deployment/       # Production app & docs
│   ├── app.py                   # Streamlit application
│   ├── setup.py
│   ├── README.md
│   └── PROJECT_REPORT.md
├── tests/                       # Automated tests
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_app.py
├── docs/                        # Additional documentation
│   ├── ARCHITECTURE.md
│   └── API_DOCUMENTATION.md
├── .github/                     # GitHub configs
│   └── workflows/
└── .gitignore
```

## Common Contribution Types

### Adding a New Model

1. Create new model in `Milestone3_ModelBuilding/`
2. Add to `model_training.py`
3. Update evaluation metrics
4. Document model in README
5. Add tests
6. Update project report

Example:
```python
# In model_training.py
from sklearn.ensemble import GradientBoostingClassifier

models_config['gradient_boost'] = (
    GradientBoostingClassifier(random_state=42),
    {'n_estimators': [100, 200], 'max_depth': [3, 5]}
)
```

### Improving Preprocessing

1. Modify `Milestone2_Preprocessing/milestone2_preprocessing.py`
2. Update preprocessing report
3. Add tests for new feature
4. Document changes
5. Test model performance impact

### Fixing a Bug

1. Create issue describing bug
2. Create feature branch: `bugfix/issue-description`
3. Fix the bug
4. Add regression test
5. Create PR referencing issue

### Adding Documentation

1. Edit relevant `.md` file or create new one
2. Follow Markdown conventions
3. Add code examples where relevant
4. Include links to related docs
5. Review for clarity

## Testing Guidelines

### Test Coverage
- Target: >80% code coverage
- All public functions tested
- Edge cases covered
- Error conditions tested

### Test Structure
```python
import pytest
from module import function

class TestFeatureName:
    def setup_method(self):
        """Setup before each test"""
        self.data = load_test_data()
    
    def test_normal_case(self):
        """Test normal operation"""
        result = function(self.data)
        assert result == expected
    
    def test_edge_case(self):
        """Test edge case"""
        result = function(edge_data)
        assert result == expected
    
    def test_error_handling(self):
        """Test error handling"""
        with pytest.raises(ValueError):
            function(invalid_data)
```

### Running Tests Locally
```bash
# All tests
pytest

# Specific test file
pytest tests/test_preprocessing.py

# Specific test function
pytest tests/test_preprocessing.py::test_missing_values_handling

# With coverage
pytest --cov=. --cov-report=html

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

## Performance Considerations

### Code Performance
- Profile code before optimizing
- Use appropriate data structures
- Minimize model inference time
- Cache computations where possible

### Scalability
- Consider dataset size growth
- Optimize for production load
- Test with realistic data volumes
- Document performance characteristics

## Documentation Standards

### Code Documentation
```python
def predict_delivery_status(order_data: dict) -> dict:
    """
    Predict on-time delivery status for an order.
    
    Args:
        order_data (dict): Order features including:
            - supplier_rating (float): 1-5
            - shipping_distance_km (int): 0-10000
            - supplier_lead_time (int): 1-30
    
    Returns:
        dict: Prediction results including:
            - prediction (str): "On-Time" or "Delayed"
            - confidence (float): 0-1
            - probabilities (dict): Model-specific probabilities
    
    Raises:
        ValueError: If order_data missing required fields
        TypeError: If data types are incorrect
    
    Example:
        >>> result = predict_delivery_status({
        ...     'supplier_rating': 4.5,
        ...     'shipping_distance_km': 500,
        ...     'supplier_lead_time': 7
        ... })
        >>> print(result['prediction'])
        "On-Time"
    """
    pass
```

### README Standards
- Clear project description
- Installation instructions
- Usage examples
- Feature list
- Contributing section
- License

### Docstring Standards
Use Google-style docstrings:
- Summary line
- Extended description (if needed)
- Args section
- Returns section
- Raises section
- Example section

## Communication

### Getting Help
- Check existing issues and discussions
- Read documentation
- Ask in GitHub Discussions
- Comment on related issues

### Asking Questions
- Be specific and clear
- Provide code examples
- Include error messages
- Describe what you've tried

### Providing Feedback
- Be constructive and respectful
- Provide specific examples
- Suggest improvements
- Acknowledge good work

## Maintenance

### Issue Management
- Respond to new issues promptly
- Ask for clarification if needed
- Reproduce bugs before fixing
- Close resolved issues
- Thank contributors

### Code Review
- Review code promptly
- Be supportive and helpful
- Suggest improvements
- Request changes when needed
- Approve good PRs

### Releases
- Update version numbers
- Create changelog
- Tag releases
- Create GitHub release
- Announce updates

## Additional Resources

- [GitHub Guides](https://guides.github.com/)
- [Git Documentation](https://git-scm.com/doc)
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Pytest Documentation](https://docs.pytest.org/)

## Recognition

We value all contributions! Contributors are recognized in:
- README.md contributors section
- Release notes
- Project discussions
- Special thanks in documentation

---

Thank you for contributing! Together we're building amazing things. 🚀

**Last Updated**: January 5, 2024
