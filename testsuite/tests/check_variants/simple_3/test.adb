package body Example3 is
   type Message is record
      null;
   end record; --place holder type
   procedure Print_Message(Msg : Message);

   type Server_Error(Error_Code : Integer) is record
      case Error_Code is
         when 500 =>
            -- Generic error
            Msg : Message;

         when 501 =>
            -- Unimplemented method
            Method : Message;

         when others =>
            null;
      end case;
   end record;

   procedure Test(Err : Server_Error) is
   begin
      case Err.Error_Code is
         when 500 =>
            Print_Message(Err.Msg);
         when 504 =>
            Print_Message(Err.Msg);
      end case;
   end Print_Error_Message;
end Example3;