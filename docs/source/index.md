OpenDelta's documentation!
=====================================

OpenDelta is a **Plug-and-play** Library of the parameter-efficient fine-tuning technology for pre-trained models.


## Essential Advantages:

1. <span style="color:orange;font-weight:bold">Clean:</span> No need to edit the backbone PTM’s codes.
2. <span style="color:green;font-weight:bold">Sustainable:</span> Most evolution in external library doesn’t require a new OpenDelta.
3. <span style="color:red;font-weight:bold">Extendable:</span> Various PTMs can share the same PET codes.
4. <span style="color:blue;font-weight:bold">Simple:</span> Applying Deltas to Huggingface examples needs as little as 2 lines of codes.
5. <span style="color:purple;font-weight:bold">Flexible:</span> Able to apply PETs to (almost) any position of the PTMs.

```{eval-rst}
.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   notes/overview.md
   notes/installation.md
   notes/usage.md
   notes/visualization.md
   notes/saveload.md

.. toctree::
   :maxdepth: 1
   :caption: Advanced Usage

   notes/keyfeature.md
   notes/autodelta.md
   notes/pluginunplug.md
   notes/acceleration.md
   notes/citation.md

.. toctree::
   :maxdepth: 2
   :caption: Package Reference

   modules/base
   modules/deltas
   modules/auto_delta


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

```