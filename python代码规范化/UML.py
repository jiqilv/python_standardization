from pyuml.diagrams import SequenceDiagram
from pyuml.tools import PlantUMLGenerator

# 创建顺序图对象
diagram = SequenceDiagram()

# 添加参与者
diagram.add_actor("RegistrationWindow")
diagram.add_actor("User")

# 添加消息
diagram.add_message("RegistrationWindow", "User", "创建User对象")
diagram.add_message("RegistrationWindow", "RegistrationWindow", "create_form()")
diagram.add_message("RegistrationWindow", "RegistrationWindow", "handleFormSubmission()")
diagram.add_message("RegistrationWindow", "RegistrationWindow", "processFormSubmission()")

# 创建 PlantUML 生成器
generator = PlantUMLGenerator()

# 生成 PlantUML 代码
plantuml_code = generator.generate(diagram)

# 保存 PlantUML 代码为文件
with open("sequence_diagram.puml", "w") as f:
    f.write(plantuml_code)
