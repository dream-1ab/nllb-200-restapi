#
 # @create date 2024-06-26 15:18:27
 # @modify date 2024-06-26 15:18:27
 # @desc [description]
#


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedTokenizer, NllbTokenizerFast
from typing import Any, Literal
import torch
import pydantic

Language = Literal['ace_Arab', 'ace_Latn', 'acm_Arab', 'acq_Arab', 'aeb_Arab', 'afr_Latn', 'ajp_Arab', 'aka_Latn', 'amh_Ethi', 'apc_Arab', 'arb_Arab', 'ars_Arab', 'ary_Arab', 'arz_Arab', 'asm_Beng', 'ast_Latn', 'awa_Deva', 'ayr_Latn', 'azb_Arab', 'azj_Latn', 'bak_Cyrl', 'bam_Latn', 'ban_Latn', 'bel_Cyrl', 'bem_Latn', 'ben_Beng', 'bho_Deva', 'bjn_Arab', 'bjn_Latn', 'bod_Tibt', 'bos_Latn', 'bug_Latn', 'bul_Cyrl', 'cat_Latn', 'ceb_Latn', 'ces_Latn', 'cjk_Latn', 'ckb_Arab', 'crh_Latn', 'cym_Latn', 'dan_Latn', 'deu_Latn', 'dik_Latn', 'dyu_Latn', 'dzo_Tibt', 'ell_Grek', 'eng_Latn', 'epo_Latn', 'est_Latn', 'eus_Latn', 'ewe_Latn', 'fao_Latn', 'pes_Arab', 'fij_Latn', 'fin_Latn', 'fon_Latn', 'fra_Latn', 'fur_Latn', 'fuv_Latn', 'gla_Latn', 'gle_Latn', 'glg_Latn', 'grn_Latn', 'guj_Gujr', 'hat_Latn', 'hau_Latn', 'heb_Hebr', 'hin_Deva', 'hne_Deva', 'hrv_Latn', 'hun_Latn', 'hye_Armn', 'ibo_Latn', 'ilo_Latn', 'ind_Latn', 'isl_Latn', 'ita_Latn', 'jav_Latn', 'jpn_Jpan', 'kab_Latn', 'kac_Latn', 'kam_Latn', 'kan_Knda', 'kas_Arab', 'kas_Deva', 'kat_Geor', 'knc_Arab', 'knc_Latn', 'kaz_Cyrl', 'kbp_Latn', 'kea_Latn', 'khm_Khmr', 'kik_Latn', 'kin_Latn', 'kir_Cyrl', 'kmb_Latn', 'kon_Latn', 'kor_Hang', 'kmr_Latn', 'lao_Laoo', 'lvs_Latn', 'lij_Latn', 'lim_Latn', 'lin_Latn', 'lit_Latn', 'lmo_Latn', 'ltg_Latn', 'ltz_Latn', 'lua_Latn', 'lug_Latn', 'luo_Latn', 'lus_Latn', 'mag_Deva', 'mai_Deva', 'mal_Mlym', 'mar_Deva', 'min_Latn', 'mkd_Cyrl', 'plt_Latn', 'mlt_Latn', 'mni_Beng', 'khk_Cyrl', 'mos_Latn', 'mri_Latn', 'zsm_Latn', 'mya_Mymr', 'nld_Latn', 'nno_Latn', 'nob_Latn', 'npi_Deva', 'nso_Latn', 'nus_Latn', 'nya_Latn', 'oci_Latn', 'gaz_Latn', 'ory_Orya', 'pag_Latn', 'pan_Guru', 'pap_Latn', 'pol_Latn', 'por_Latn', 'prs_Arab', 'pbt_Arab', 'quy_Latn', 'ron_Latn', 'run_Latn', 'rus_Cyrl', 'sag_Latn', 'san_Deva', 'sat_Beng', 'scn_Latn', 'shn_Mymr', 'sin_Sinh', 'slk_Latn', 'slv_Latn', 'smo_Latn', 'sna_Latn', 'snd_Arab', 'som_Latn', 'sot_Latn', 'spa_Latn', 'als_Latn', 'srd_Latn', 'srp_Cyrl', 'ssw_Latn', 'sun_Latn', 'swe_Latn', 'swh_Latn', 'szl_Latn', 'tam_Taml', 'tat_Cyrl', 'tel_Telu', 'tgk_Cyrl', 'tgl_Latn', 'tha_Thai', 'tir_Ethi', 'taq_Latn', 'taq_Tfng', 'tpi_Latn', 'tsn_Latn', 'tso_Latn', 'tuk_Latn', 'tum_Latn', 'tur_Latn', 'twi_Latn', 'tzm_Tfng', 'uig_Arab', 'ukr_Cyrl', 'umb_Latn', 'urd_Arab', 'uzn_Latn', 'vec_Latn', 'vie_Latn', 'war_Latn', 'wol_Latn', 'xho_Latn', 'ydd_Hebr', 'yor_Latn', 'yue_Hant', 'zho_Hans', 'zho_Hant', 'zul_Latn']
language_list = pydantic.TypeAdapter[Language](Language).json_schema()["enum"]

class Translator:
    source_tokenizer: NllbTokenizerFast
    destination_tokenizer: NllbTokenizerFast
    model: Any
    model_name: str
    source_language: str
    destination_language: str
    device: Literal["cpu", "cuda:0"]
    
    def __init__(self, model: Literal["600m", "1.3b", "3.3b", "54b", "or_your_custom_nllb_model_from_huggingface, e.g. facebook/nllb-200-distilled-600M"] = "600m"):
        model_name_pairs = {
            "600m": "facebook/nllb-200-distilled-600M",
            "1.3b": "facebook/nllb-200-1.3B",
            "3.3b": "facebook/nllb-200-3.3B",
            "54b": "facebook/nllb-moe-54b"
        }
        self.model_name = model_name_pairs.get(model)
        if self.model_name == None:
            self.model_name = model
            
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        model_name = self.model_name
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
    def set_direction(self, source: Language, destination: Language):
        self.source_tokenizer = AutoTokenizer.from_pretrained(self.model_name, src_lang=source)
        self.destination_tokenizer = AutoTokenizer.from_pretrained(self.model_name, src_lang=destination)
        self.source_language = source
        self.destination_language = destination
    def swap_direction(self):
        temp = self.source_language
        self.source_language = self.destination_language
        self.destination_language = temp

        temp = self.source_tokenizer
        self.source_tokenizer = self.destination_tokenizer
        self.destination_tokenizer = temp
        
    def translate(self, text: str) -> str:
        source_tokenizer = self.source_tokenizer
        token_of_input = source_tokenizer(text=text, return_tensors="pt").to(self.device)
        translated_token = self.model.generate(**token_of_input, max_length=3000, forced_bos_token_id=self.source_tokenizer.lang_code_to_id[self.destination_language]).to(self.device)
        translated_text = source_tokenizer.batch_decode(translated_token, skip_special_tokens=True)[0]
        return translated_text
