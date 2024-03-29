## Description ##
(Brief description on what this PR is about)

## Checklist ##
### Essentials ###
Please feel free to remove inapplicable items for your PR.
- [ ] The issue title and its ID
- [ ] Changes are complete (i.e. I finished coding on this PR)
- [ ] All changes have test coverage:
- Unit tests are added for small changes to verify correctness (e.g. adding a new operator)
- Nightly tests are added for complicated/long-running ones (e.g. changing distributed kvstore)
- Build tests will be added for build configuration changes (e.g. adding a new build option with NCCL)
- [ ] Code is well-documented: 
- For user-facing API changes, API doc string has been updated.
- For new examples, README.md is added to explain the what the example does, the source of the dataset, expected performance on test set and reference to the original paper if applicable
- Check the API doc at http://mxnet-ci-doc.s3-accelerate.dualstack.amazonaws.com/PR-$PR_ID/$BUILD_ID/index.html
- [ ] To the my best knowledge, examples are either not affected by this change, or have been fixed to be compatible with this change

### Changes ###
- [ ] Feature1, tests, (and when applicable, API doc)
- [ ] Feature2, tests, (and when applicable, API doc)

## Comments ##
- If this change is a backward incompatible change, why must this change be made.
- Interesting edge cases to note here
