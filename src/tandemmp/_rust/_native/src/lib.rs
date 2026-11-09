//! The `tandemmp._rust._native` cdylib: registers each sibling crate's
//! PyO3 module as a submodule of `tandemmp._rust`, so e.g.
//! `from tandemmp._rust import drift_correction` keeps working even though
//! the compiled extension is a single shared library. `sys.modules`
//! registration (beyond `add_submodule`'s attribute) is needed for direct
//! dotted imports (`import tandemmp._rust.drift_correction`) and for
//! pickling to resolve `__module__` correctly.
//!
//! Add future crates (e.g. `psf`) the same way: create the submodule,
//! call its registration function, `add_submodule`, and register it in
//! `sys.modules` under `tandemmp._rust.<name>`.

use pyo3::prelude::*;
use pyo3::types::PyModule;

#[pymodule]
fn _native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let drift_correction_mod = PyModule::new(py, "drift_correction")?;
    drift_correction::python::drift_correction(&drift_correction_mod)?;
    m.add_submodule(&drift_correction_mod)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("tandemmp._rust.drift_correction", &drift_correction_mod)?;
    Ok(())
}
