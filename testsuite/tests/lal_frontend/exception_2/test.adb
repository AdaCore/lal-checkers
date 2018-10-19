procedure Test is
   A_Failure : exception;
   B_Failure : exception;

   X : Integer;
begin
   X := 3;
   declare
      Y : Integer := X;
   begin
      Y := Y + 1;
   exception
      when A_Failure =>
         X := 41;
      when B_Failure =>
         X := 59;
   end;
end Test;
