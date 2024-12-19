from typing import List, Optional

from pydantic import BaseModel


class ProcessingSoftware(BaseModel):
    name: str
    version: str


# for make_minislabs command
class ProcessingInputMakeMinislabs(BaseModel):
    in_coords: str
    in_vol: Optional[str]


class ProcessingOutputMakeMinislabs(BaseModel):
    out_dir: str


class ProcessingParametersMakeMinislabs(BaseModel):
    extract_shape: List[int]
    voxel_spacing: float
    coords_scale: float
    col_name: Optional[str]
    tomo_type: Optional[str]
    user_id: Optional[str]
    session_id: Optional[str]
    particle_name: Optional[str]
    angles: List[int]
    gallery_shape: List[int]
    live: bool
    t_interval: float
    t_exit: float


class ProcessingConfigMakeMinislabs(BaseModel):
    software: ProcessingSoftware
    input: ProcessingInputMakeMinislabs
    output: ProcessingOutputMakeMinislabs
    parameters: ProcessingParametersMakeMinislabs


# for cs_center_picks command
class ProcessingInputCsCenterPicks(BaseModel):
    cs_file: str
    map_file: str


class ProcessingOutputCsCenterPicks(BaseModel):
    cs_file: str


class ProcessingParametersCsCenterPicks(BaseModel):
    gallery_shape: List[int]


class ProcessingConfigCsCenterPicks(BaseModel):
    software: ProcessingSoftware
    input: ProcessingInputCsCenterPicks
    output: ProcessingOutputCsCenterPicks
    parameters: ProcessingParametersCsCenterPicks


# for cs_map_particles command
class ProcessingInputCsMapParticles(BaseModel):
    copick_json: Optional[str]
    in_star: Optional[str]
    in_star_multiple: Optional[List]
    cs_file: str
    map_file: str


class ProcessingOutputCsMapParticles(BaseModel):
    out_file: str


class ProcessingParametersCsMapParticles(BaseModel):
    col_name: Optional[str]
    particle_name: Optional[str]
    session_id: Optional[str]
    user_id: Optional[str]
    coords_scale: float
    apix: float
    rejected_set: bool


class ProcessingConfigCsMapParticles(BaseModel):
    software: ProcessingSoftware
    input: ProcessingInputCsMapParticles
    output: ProcessingOutputCsMapParticles
    parameters: ProcessingParametersCsMapParticles


# for rln_map_particles command
class ProcessingInputRlnMapParticles(BaseModel):
    rln_file: str
    map_file: str
    coords_file: str

    
class ProcessingOutputRlnMapParticles(BaseModel):
    out_file: str

    
class ProcessingParametersRlnMapParticles(BaseModel):
    particle_name: str
    session_id: Optional[str]
    user_id: Optional[str]
    apix: Optional[float]
    session_id_out: str
    user_id_out: str

    
class ProcessingConfigRlnMapParticles(BaseModel):
    software: ProcessingSoftware
    input: ProcessingInputRlnMapParticles
    output: ProcessingOutputRlnMapParticles
    parameters: ProcessingParametersRlnMapParticles
