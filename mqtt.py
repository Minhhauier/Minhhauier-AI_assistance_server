import paho.mqtt.client as mqtt
import json
import time
import ssl
import parameter_secret
PORT = 8883
PUB_TOPIC = "component/sub"

class MQTTController:
    def __init__(self, broker=parameter_secret.MQTT_BROKER, port=8883):
        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.username_pw_set(parameter_secret.USER_NAME, parameter_secret.PASS_WORD)
        # HiveMQ Cloud on port 8883 requires TLS.
        self.client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
        self.client.reconnect_delay_set(min_delay=1, max_delay=10)
        self.client.connect(broker, port, keepalive=60)
        self.client.loop_start()
        time.sleep(0.5)

    def _on_connect(self, client, userdata, flags, rc):
        print("[MQTT] Connected" if rc == 0 else f"[MQTT] Failed rc={rc}")

    def _on_disconnect(self, client, userdata, rc):
        print(f"[MQTT] Disconnected rc={rc}")

    def publish_control(self, cmd_type: int, type_: int, status: int, topic=PUB_TOPIC):
        payload = json.dumps({"command_type": cmd_type, "data": {"type": type_, "status": status}}, indent=2)
        if not self.client.is_connected():
            print("[MQTT] Chưa kết nối broker, thử reconnect...")
            try:
                self.client.reconnect()
                time.sleep(0.2)
            except Exception as e:
                print(f"[MQTT] Reconnect lỗi: {e}")
                return

        info = self.client.publish(topic, payload, qos=1)
        info.wait_for_publish()

        if info.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"[MQTT] Sent to {topic}: {payload}")
        else:
            print(f"[MQTT] Publish failed rc={info.rc}: {mqtt.error_string(info.rc)}")

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()

# Singleton — khởi tạo 1 lần khi import
_instance = MQTTController()

def publish_control(cmd_type: int, type_: int, status: int):
    _instance.publish_control(cmd_type, type_, status)