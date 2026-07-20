"""Lightweight, serializable adapter for Google TabFM's JAX classifier."""

from tabfm import TabFMClassifier, tabfm_v1_0_0_jax


def load_tabfm_model():
    return tabfm_v1_0_0_jax.load(
        model_type="classification",
        col_attention_impl="jax",
        row_attention_impl="jax",
        icl_attention_impl="jax",
    )


class LazyTabFMClassifier(TabFMClassifier):
    """Exclude multi-GB pretrained weights from joblib artifacts and reload lazily."""

    def __getstate__(self):
        state = self.__dict__.copy()
        state["model"] = None
        state.pop("_predict_step_compiled_with_cat", None)
        state.pop("_predict_step_compiled_no_cat", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _ensure_model(self):
        if self.model is None:
            self.model = load_tabfm_model()

    def predict(self, X):
        self._ensure_model()
        return super().predict(X)

    def predict_proba(self, X):
        self._ensure_model()
        return super().predict_proba(X)
