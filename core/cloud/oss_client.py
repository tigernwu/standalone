import oss2
import time
import random
from itertools import islice
from typing import Optional, Tuple, Union, List
from core.utils.config_setting import Config
from core.utils.log import logger

class OSSClient:
    def __init__(self, endpoint: str = "https://oss-cn-beijing.aliyuncs.com", 
                 region: str = "cn-beijing",
                 bucket_name: Optional[str] = "tumedia"):
        """
        Initialize OSS client with credentials from config
        
        Args:
            endpoint: OSS endpoint URL
            region: OSS region (e.g. cn-hangzhou)
            bucket_name: Optional bucket name. If not provided, generates a unique name
        """
        
        # Get credentials from config
        config = Config()
        access_key_id = config.get("OSS_ACCESS_KEY_ID")
        access_key_secret = config.get("OSS_ACCESS_KEY_SECRET")
        
        if not all([access_key_id, access_key_secret]):
            raise ValueError("Missing required OSS credentials in config")

        # Set up authentication
        self.auth = oss2.AuthV4(access_key_id, access_key_secret)

        # Remove 'https://' from endpoint for URL generation
        self.raw_endpoint = endpoint.replace('https://', '').replace('http://', '')
        self.endpoint = endpoint
        self.region = region
        self.bucket_name = bucket_name or self._generate_unique_bucket_name()
        self.bucket = oss2.Bucket(self.auth, self.endpoint, self.bucket_name, region=self.region)

    def _generate_unique_bucket_name(self) -> str:
        """Generate a unique bucket name using timestamp and random number"""
        timestamp = int(time.time())
        random_number = random.randint(0, 9999)
        return f"demo-{timestamp}-{random_number}"

    def _generate_file_url(self, object_name: str) -> str:
        """
        Generate the public URL for an uploaded file
        
        Args:
            object_name: Name of the object in OSS
            
        Returns:
            str: Public URL of the file
        """
        return f"https://{self.bucket_name}.{self.raw_endpoint}/{object_name}"

    def create_bucket(self, acl: str = oss2.models.BUCKET_ACL_PRIVATE) -> bool:
        """
        Create a new bucket with specified ACL
        
        Args:
            acl: Access control level for the bucket
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.bucket.create_bucket(acl)
            logger.info(f"Bucket {self.bucket_name} created successfully")
            return True
        except oss2.exceptions.OssError as e:
            logger.error(f"Failed to create bucket: {e}")
            return False

    def upload_string(self, object_name: str, content: str, headers: dict = None) -> Optional[oss2.models.PutObjectResult]:
        """
        Upload a string to OSS
        
        Args:
            object_name: Name of the object in OSS
            content: String content to upload
            headers: Optional headers for the upload
            
        Returns:
            PutObjectResult if successful, None otherwise
        """
        try:
            result = self.bucket.put_object(object_name, content, headers=headers)
            logger.info(f"String uploaded successfully to {object_name}")
            return result
        except oss2.exceptions.OssError as e:
            logger.error(f"Failed to upload string: {e}")
            return None

    def upload_file(self, object_name: str, file_path: str) -> Tuple[Optional[oss2.models.PutObjectResult], Optional[str]]:
        """
        Upload a local file to OSS and return the upload result and public URL
        
        Args:
            object_name: Name of the object in OSS
            file_path: Local file path
            
        Returns:
            Tuple containing:
            - PutObjectResult if successful, None otherwise
            - Public URL of the uploaded file if successful, None otherwise
        """
        try:
            result = self.bucket.put_object_from_file(object_name, file_path)
            if result.status == 200:
                file_url = self._generate_file_url(object_name)
                logger.info(f"File uploaded successfully to {object_name}")
                logger.info(f"File URL: {file_url}")
                return result, file_url
            return result, None
        except oss2.exceptions.OssError as e:
            logger.error(f"Failed to upload file: {e}")
            return None, None

    def append_object(self, object_name: str, position: int, content: str, 
                     headers: dict = None) -> Optional[oss2.models.AppendObjectResult]:
        """
        Append content to an object
        
        Args:
            object_name: Name of the object in OSS
            position: Position to append from
            content: Content to append
            headers: Optional headers for the append operation
            
        Returns:
            AppendObjectResult if successful, None otherwise
        """
        try:
            result = self.bucket.append_object(object_name, position, content, headers=headers)
            logger.info(f"Content appended successfully to {object_name}")
            return result
        except oss2.exceptions.OssError as e:
            logger.error(f"Failed to append content: {e}")
            return None

    def download_file(self, object_name: str, file_path: str) -> bool:
        """
        Download an object to a local file
        
        Args:
            object_name: Name of the object in OSS
            file_path: Local file path to save to
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.bucket.get_object_to_file(object_name, file_path)
            logger.info(f"File downloaded successfully to {file_path}")
            return True
        except oss2.exceptions.OssError as e:
            logger.error(f"Failed to download file: {e}")
            return False

    def object_exists(self, object_name: str) -> bool:
        """
        Check if an object exists
        
        Args:
            object_name: Name of the object in OSS
            
        Returns:
            bool: True if exists, False otherwise
        """
        try:
            return self.bucket.object_exists(object_name)
        except oss2.exceptions.OssError as e:
            logger.error(f"Failed to check object existence: {e}")
            return False

    def list_objects(self, prefix: str = '', max_objects: int = 100) -> List[str]:
        """
        List objects in the bucket
        
        Args:
            prefix: Prefix to filter objects
            max_objects: Maximum number of objects to list
            
        Returns:
            List of object keys
        """
        try:
            objects = list(islice(oss2.ObjectIterator(self.bucket, prefix=prefix), max_objects))
            return [obj.key for obj in objects]
        except oss2.exceptions.OssError as e:
            logger.error(f"Failed to list objects: {e}")
            return []

    def delete_object(self, object_name: str) -> bool:
        """
        Delete an object
        
        Args:
            object_name: Name of the object to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.bucket.delete_object(object_name)
            logger.info(f"Object {object_name} deleted successfully")
            return True
        except oss2.exceptions.OssError as e:
            logger.error(f"Failed to delete object: {e}")
            return False

    def delete_bucket(self) -> bool:
        """
        Delete the bucket and all its contents
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Delete all objects first
            objects = self.list_objects()
            for obj in objects:
                self.delete_object(obj)
            
            # Delete the bucket
            self.bucket.delete_bucket()
            logger.info(f"Bucket {self.bucket_name} deleted successfully")
            return True
        except oss2.exceptions.OssError as e:
            logger.error(f"Failed to delete bucket: {e}")
            return False