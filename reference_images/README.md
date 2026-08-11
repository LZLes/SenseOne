# reference_images/

Your electrode photos go here, one subfolder per batch (fabrication
date), e.g. `reference_images/20260804/`. Not shipped with the repo
(private/large) -- this folder is empty until you add your own.

Filenames need to encode a batch/sheet/electrode identity the agent
can parse -- see `tools/image_qc.py:_parse_filename_identity` for the
conventions already handled, or describe your own naming to the agent
and it'll help adapt the parser.

See the main README's "Electrode photos" section for how these get
referenced in conversation once populated.
