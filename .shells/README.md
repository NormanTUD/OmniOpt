# What is this?

This allows you to do easy autocompletion for lots of the OmniOpt2 programs in different shells (zsh, bash).

## Installation

```
THIS_SHELL=$(echo $SHELL | sed -e 's#.*/##')
if [[ -e  "${THIS_SHELL}_omniopt_compdefs" ]]; then
    cp "${THIS_SHELL}_omniopt_compdefs" "$HOME/.${THIS_SHELL}_omniopt_compdefs"
    grep -qxF "source $HOME/.${THIS_SHELL}_omniopt_compdefs" "$HOME/.${THIS_SHELL}rc" || echo "source $HOME/.${THIS_SHELL}_omniopt_compdefs" >> "$HOME/.${THIS_SHELL}rc"
    source ~/.${THIS_SHELL}rc
else
    echo "${THIS_SHELL}_omniopt_compdefs not found for shell $THIS_SHELL"
fi
```
