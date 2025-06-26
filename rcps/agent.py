# rcps/agent.py (企业级重构版)
import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateError
from typing import Dict, Any, Optional, Union, List
import os
from .llms import BaseLLM
from .exceptions import ConfigError
from .utils import get_logger, package_join

logger = get_logger(__name__)


class Agent:
    def __init__(
            self,
            role_name: str,
            system_prompt: str,
            template: "jinja2.Template",
            llm: BaseLLM,
            return_json: bool,
    ):
        self.name = role_name
        self.system_prompt = system_prompt
        self.template = template
        self.llm = llm
        self.return_json = return_json
        logger.info(f"Agent '{self.name}' initialized.")

    def execute(self, **context) -> Union[str, Dict, List]:

        try:
            prompt = self.template.render(**context)
        except TemplateError as e:
            logger.error(f"Failed to render prompt for agent '{self.name}': {e}")
            raise ConfigError(f"Prompt rendering error for '{self.name}': {e}") from e

        return self.llm.execute(
            prompt,
            system_message=self.system_prompt,
            return_json=self.return_json
        )

    async def execute_async(self, **context) -> Union[str, Dict, List]:
        """
        异步执行Agent任务。
        """
        try:
            prompt = self.template.render(**context)
        except TemplateError as e:
            logger.error(f"Failed to render prompt for agent '{self.name}': {e}")
            raise ConfigError(f"Prompt rendering error for '{self.name}': {e}") from e

        return await self.llm.execute_async(
            prompt,
            system_message=self.system_prompt,
            return_json=self.return_json
        )


class AgentFactory:
    def __init__(self, llm_map: Dict[str, BaseLLM]):
        """
        Args:
            llm_map (Dict[str, BaseLLM]): 一个将LLM名称映射到LLM实例的字典。
        """
        self.llm_map = llm_map
        self.roles_dir = package_join("rcps", "roles")
        self.prompts_dir = package_join("rcps", "prompts")
        self._agents: Dict[str, Agent] = {}

        # 配置Jinja2环境
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.prompts_dir),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True
        )
        logger.info(f"AgentFactory initialized. Loading roles from: {self.roles_dir}")

    def get_agent(self, role_name: str) -> Agent:
        if role_name not in self._agents:
            self._agents[role_name] = self._create_agent(role_name)
        return self._agents[role_name]

    def _create_agent(self, role_name: str) -> Agent:
        """
        根据配置文件创建一个新的Agent实例。
        """
        role_config_path = os.path.join(self.roles_dir, f"{role_name}.yaml")
        prompt_template_path = f"{role_name}_prompt.txt"

        try:
            with open(role_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            raise ConfigError(f"Role configuration file not found: {role_config_path}")
        except yaml.YAMLError as e:
            raise ConfigError(f"Error parsing role configuration file {role_config_path}: {e}")

        # 验证配置
        required_keys = ['system_prompt', 'use_llm', 'prompt_template']
        if not all(key in config for key in required_keys):
            raise ConfigError(f"Role config '{role_name}.yaml' is missing one of the required keys: {required_keys}")

        # 获取LLM实例
        llm_key = config['use_llm']
        llm_instance = self.llm_map.get(llm_key)
        if not llm_instance:
            raise ConfigError(f"LLM key '{llm_key}' specified in '{role_name}.yaml' not found in provided LLM map.")

        # 加载Prompt模板
        try:
            template = self.jinja_env.get_template(config['prompt_template'])
        except TemplateError as e:
            raise ConfigError(
                f"Failed to load prompt template '{config['prompt_template']}' for role '{role_name}': {e}")

        return Agent(
            role_name=role_name,
            system_prompt=config['system_prompt'],
            template=template,
            llm=llm_instance,
            return_json=config.get('return_json', False)
        )