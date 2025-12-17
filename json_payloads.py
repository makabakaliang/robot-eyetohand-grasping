class QueryPayloadFactory:
    """查询请求JSON负载工厂类"""

    @staticmethod
    def _create_payload(query_addr, pack_id="0"):
        """创建通用查询负载"""
        return {
            "dsID": "www.hc-system.com.RemoteMonitor",
            "reqType": "query",
            "packID": pack_id,
            "queryAddr": query_addr
        }
    
    @staticmethod
    def create_command(cmd_data, pack_id="0"):
        """创建通用命令负载"""
        return {
            "dsID": "www.hc-system.com.RemoteMonitor",
            "reqType": "command",
            "packID": pack_id,
            "cmdData": cmd_data
        }

    # n从0开始，0-5，J1到J6.
    @staticmethod
    def create_joint_angle_payload(pack_id="0"):
        """创建关节角度查询负载"""
        # 修正笔误：axis6-3 → axis-3
        query_addr = ["axis-0", "axis-1", "axis-2", "axis-3", "axis-4", "axis-5"]
        return QueryPayloadFactory._create_payload(query_addr, pack_id)

    # n从0开始，0-5，x,y,z,u,v,w.
    @staticmethod
    def create_world_coord_payload(pack_id="0"):
        """创建世界坐标查询负载"""
        query_addr = ["world-0", "world-1", "world-2", "world-3", "world-4", "world-5"]
        return QueryPayloadFactory._create_payload(query_addr, pack_id)
    
    # 从0开始，0时无报警
    @staticmethod
    def create_robot_curAlarm(pack_id="0"):
        """创建机器人警报状态查询"""
        query_addr = ["curAlarm"]
        return QueryPayloadFactory._create_payload(query_addr, pack_id)
    
    # 0无，1手动模式，2自动模式，3停止模式，7自动运行中，8单步，9单循环
    @staticmethod
    def create_robot_model(pack_id="0"):
        """创建机器人模式查询"""
        query_addr = ["curMode"]
        return QueryPayloadFactory._create_payload(query_addr, pack_id)
    
    # 立即停止当前动作（使能断开，重新上使能，启动重头开始）
    @staticmethod
    def create_robot_stop(pack_id="0"):
        """创建机器人停止命令负载"""
        cmd_data = ["actionStop","r1","r2"]
        # 修正：命令类型应调用create_command而非_create_payload
        return QueryPayloadFactory.create_command(cmd_data, pack_id)
    
    # 暂停当前动作（启动从当前步开始）
    @staticmethod
    def create_robot_pause(pack_id="0"):
        """创建机器人暂停命令负载"""
        cmd_data = ["actionPause","r1","r2"]
        return QueryPayloadFactory.create_command(cmd_data, pack_id)
    
    # 启动（修复重复方法名问题）
    @staticmethod
    def create_robot_start(pack_id="0"):
        """创建机器人启动命令负载"""
        cmd_data = ["startButton","r1","r2"]
        return QueryPayloadFactory.create_command(cmd_data, pack_id)
    
    # 停止按键，消除警报
    @staticmethod
    def create_robot_stopButton(pack_id="0"):
        """创建机器人消除警报命令负载"""
        cmd_data = ["stopButton","r1","r2"]
        return QueryPayloadFactory.create_command(cmd_data, pack_id)    

    # 创建机器人拍照成功反馈负载   
    @staticmethod
    def create_photo_payload(cam_id=0, ret=1):
        """创建机器人拍照成功反馈负载"""
        return {
            "dsID": "www.hc-system.com.cam",
            "reqType":"photo", 
            "camID":cam_id,
            "ret":ret
        }

    # 创建机器人拍照运动命令负载
    @staticmethod
    def create_add_points_payload(
        pack_id="0",
        empty_list="1",
        cam_id="0",
        points=None
    ):
        """
        创建添加点的请求负载
        :param pack_id: 包ID，默认"0"
        :param empty_list: 是否清空列表，默认"1"
        :param cam_id: 相机ID，默认"0"
        :param points: 点数据列表，每个元素包含ModelID、X、Y等字段
        """
        # 设置默认点数据
        if points is None:
            points = [
                {
                    "ModelID": "0",
                    "X": "0.0",
                    "Y": "0.0",
                    "Z": "0.0",
                    "U": "0",
                    "V": "0",
                    "Angle": "0",
                    "Similarity": "0",
                    "Color": "0",
                    "Rel": "0"
                }
            ]
        
        # 构建dsData结构
        ds_data = [
            {
                "camID": cam_id,
                "data": points
            }
        ]
        
        # 构建完整负载（补充packID字段，与test1逻辑对齐）
        return {
            "dsID": "www.hc-system.com.cam",
            "emptyList": empty_list,
            "reqType": "AddPoints",
            "packID": pack_id,
            "dsData": ds_data
        }

    # 创建机器人运动命令负载
    @staticmethod
    def create_sport_payload(
        pack_id="0",
        empty_list="1",#是否清空远程列表
        oneshot="0",#1执行一次，0一直执行
        action="10",#4自由路径，10姿势直线，17姿势曲线
        m_values=None,
        ck_status="0x3F",#0x3F六轴，0xFF八轴
        speed="80.0",#速度，精度0.1%
        delay="1.0",#动作前延时，精度0.1秒
        coord="0",#工作台编号
        tool="0",#工具编号
        smooth="0"#平滑等级，0-9
    ):
        """
        创建动作指令 JSON
        :param m_values: 列表或元组，共8个字符串，表示 m0 到 m7 的值
        """
        if m_values is None:
            m_values = ["0.000"] * 8  # 默认 8 个关节为 0.000

        # 构造 instruction 字典
        instruction = {
            "oneshot": oneshot,
            "action": action,
            "ckStatus": ck_status,
            "speed": speed,
            "delay": delay,
            "coord": coord,
            "tool": tool,
            "smooth": smooth
        }

        # 添加 m0 ~ m7
        for i in range(8):
            instruction[f"m{i}"] = m_values[i]

        # 构造完整 payload
        return {
            "dsID": "www.hc-system.com.HCRemoteCommand",
            "reqType": "AddRCC",
            "emptyList": empty_list,
            "packID": pack_id,
            "instructions": [instruction]
        }

# 使用示例 
# if __name__ == "__main__":
#     # 自定义角度：让 m4 = -90，其余为 0
#     payload2 = QueryPayloadFactory.create_sport_payload(
#         pack_id="1234",
#         m_values=["0.000", "0.000", "0.000", "0.000", "-90.000", "0.000", "0.000", "0.000"],
#         speed="100.0",
#         delay="0.5"
#     )
#     print(json.dumps(payload2, indent=2))